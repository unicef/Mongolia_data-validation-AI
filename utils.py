from pyproj import Proj, transform
from shapely.geometry import (
    box, 
    Polygon, mapping # for pre-processing OSM data
)
from sklearn.cluster import DBSCAN
import geopandas as gpd
import pandas as pd
import numpy as np

from tqdm import tqdm
import os



def dissolve_geometry(df):
    """
    Merge overlapping geometries from a GeoDataFrame.
    Example use case: Used in merging overlapping polygons from a single dataframe. 
                      Combining amenity polygons and building polygons from OSM.
    
    Reference: https://gis.stackexchange.com/questions/334459/how-to-dissolve-overlapping-polygons-using-geopandas
    
    Args:
        df (geopandas GeoDataFrame): GeoDataFrame containing overlapping geometries to be merged.
        
    Returns:
        dissolved_df (geopandas GeoDataFrame): a new GeoDataFrame with overlapping geometries merged.
    """
    
    src_df = df.copy()

    geoms = src_df.geometry.unary_union

    dissolved_df = gpd.GeoDataFrame(geometry=[geoms], crs = src_df.crs)

    dissolved_df = dissolved_df.explode(index_parts=True).reset_index(drop=True)

    return dissolved_df


def convert_linestrings_to_polygons(df):
    """
    Converts linestrings from a GeoDataFrame to polygons.
    Example use case: Converts linestring data from OSM to polygons to merge them with other polygon data from OSM.
    
    Reference: https://stackoverflow.com/questions/2964751/how-to-convert-a-geos-multilinestring-to-polygon
    
    Args:
        df (geopandas GeoDataFrame): GeoDataFrame containing linestrings (to be converted to polygons).
        
    Returns:
        new_df (geopandas GeoDataFrame): a new GeoDataFrame with polygons (converted from linestrings).
    """
    
    src_df = df.copy()

    new_df = gpd.GeoDataFrame(
        geometry= [Polygon(mapping(x)['coordinates']) for x in src_df.geometry],
        crs = src_df.crs
    )

    return new_df


def convert_polygons_to_points(df):
    """
    Converts polygons from input GeoDataFrame to points.
    Example use case: Used for converting polygons from OSM to points for downloading satellite imagery.
    
    Reference: https://gis.stackexchange.com/questions/372564/userwarning-when-trying-to-get-centroid-from-a-polygon-geopandas
    
    Args:
        df (geopandas GeoDataFrame): GeoDataFrame containing polygons (to be converted to points).
        
    Returns:
        new_df (geopandas GeoDataFrame): a new GeoDataFrame with points (converted from polygons).
    """

    # CRS of input GeoDataFrame
    src_crs = df.crs

    new_df = df.copy()
    
    # convert to EPSG:3857 first to create centroids (use a CRS with coordinates in meters)
    new_df = new_df.to_crs("EPSG:3857")

    # add centroids to current GeoDataFrame
    new_df["centroids"] = new_df.centroid

    # keep only "centroids" column
    new_df = new_df[['centroids']]

    # rename column from "centroids" to "geometry"
    new_df.rename(columns={'centroids':'geometry'}, inplace=True)

    # set geometry
    new_df = new_df.set_geometry("geometry")

    # convert CRS back to original DataFrame CRS
    new_df = new_df.to_crs(src_crs)
    
    return new_df



####################  Creating extent  ####################

def create_extent_from_centroid(src_crs, x, y, grid_width, grid_height, spatial_resolution):
    """
    Create top, left, bottom, right points from center points.
    First converts CRS to "EPSG:3857" and calculate four corner points by actual ground distance in meters.
    And converts back to original CRS.
    
    Args:
        src_crs (str) : coordinate reference system. e.g. "EPSG:4326"
        x (pandas Series) : longitude coordinates
        y (pandas Series) : latitude coordinates
        grid_width (int) : image pixel size. e.g. 512
        grid_height (int) : image pixel size. e.g. 512
        spatial_resolution(float) : ground distance in meters. e.g. 0.6 m
    
    Returns:
        a tuple of multiple pandas Series: (top, left, bottom, right)
    """
    
    crs_epsg_3857 = "EPSG:3857"
    
    # for changing crs
    src_Proj, epsg_3857_Proj = Proj(init=src_crs), Proj(init=crs_epsg_3857)

    # change crs to "EPSG:3857"
    x, y = transform(src_Proj, epsg_3857_Proj, x, y)
    
    # calculate four corner points from centroids
    top = y + (grid_height / 2) * spatial_resolution
    bottom = y - (grid_height / 2) * spatial_resolution
    left = x - (grid_width / 2) * spatial_resolution
    right = x + (grid_width / 2) * spatial_resolution
    
    # change crs back to src_crs
    left, top = transform(epsg_3857_Proj, src_Proj, left, top)
    right, bottom = transform(epsg_3857_Proj, src_Proj, right, bottom)
    
    return top, left, bottom, right


def produce_geojson(df, crs, columns, save_dir):
    """
    Produce a geojson file with grid polygons using top, left, bottom, right points from a pandas DataFrame.
    
    Args:
        df (pandas DataFrame): DataFrame containing four corner grid points (top, left, bottom, right)
        crs (str): crs of the grid corner points. e.g. "EPSG:4326", "EPSG:3857"
        columns (list): columns from DataFrame that will be included in geojson file. e.g. ["image_id", "lat", "lon", "top", "left", "bottom", "right"]
        save_dir (str): file path where geojson file will be saved
        
    Returns:
        None
    """
    
    # create grid polygons
    x_min_list = list(df["left"])
    y_min_list = list(df["bottom"])
    x_max_list = list(df["right"])
    y_max_list = list(df["top"])

    grids_list = []

    for xmin, ymin, xmax, ymax in zip(x_min_list, y_min_list, x_max_list, y_max_list):
        grid = box(xmin, ymin, xmax, ymax)
        grids_list.append(grid)
        
    # create GeoDataFrame
    geo_df = gpd.GeoDataFrame(df[columns], 
                              geometry = grids_list,
                              crs = crs)
    
    # save geojson file
    geo_df.to_file(save_dir)



####################  Add model prediction probability scores to dataframe  ####################

def add_yolov5_conf_scores(original_df, predictions_folder, school_conf_col = "conf_yolov5", image_id_col = "image_id"):
    """
    Get probability scores for school from YOLOv5 model prediction text files and add them to DataFrame.
    Text files contains predictions in (class, x, y, w, h, conf) format.
    
    Args:
        original_df (pandas DataFrame): YOLOv5 probability scores will be added to the DataFrame
        predictions_folder (str): folder containing YOLOv5 model prediction text files
        school_conf_col (str): column name for YOLOv5 probability scores in DataFrame
        image_id_col (str): column name for image ids in DataFrame
        
    Returns:
        DataFrame with YOLOv5 model probability scores added in a separate column
    """
    
    df = original_df.copy()

    # parse YOLOv5 prediction text files
    pred_text_list = os.listdir(predictions_folder)

    # loop through each prediction text file
    for filename in tqdm(pred_text_list):

        text_dir = os.path.join(predictions_folder, filename)

        with open(text_dir) as file:
            data = file.readlines()
            conf_list = []

            for line in data:
                line = line.strip()
                conf = float(line.split(' ')[-1])
                conf_list.append(conf)

            # multiple bbox could be detected, take the max conf score
            school_conf = max(conf_list)  

        # add current image prediction conf score to DataFrame
        df.loc[ df[image_id_col] == filename.replace('.txt', ''), 
                school_conf_col] = school_conf

    return df


def add_efficientnet_conf_scores(original_df, predictions_folder, school_class_id = 1, school_conf_col = "conf_efficientnet", image_id_col = "image_id"):
    """
    Get probability scores for school from EfficientNet model prediction text files and add them to DataFrame.
    Text files contains predictions in (non_school_prob, school_prob) format.
    School class id is 1.
    
    Args:
        original_df (pandas DataFrame): EfficientNet probability scores will be added to this DataFrame
        predictions_folder (str): folder containing EfficientNet model prediction text files
        school_conf_col (str): column name for EfficientNet probability scores in DataFrame
        image_id_col (str): column name for image ids in DataFrame
        
    Returns:
        DataFrame with EfficientNet model probability scores added in a separate column
    """
    
    df = original_df.copy()
    
    pred_text_list = os.listdir(predictions_folder)

    # loop through each prediction text file
    for filename in tqdm(pred_text_list):

        text_dir = os.path.join(predictions_folder, filename)
        
        with open(text_dir) as file:
            data = file.readlines()[0]
            school_conf = data.split(' ')[school_class_id]
            school_conf = float(school_conf)

        df.loc[ df[image_id_col] == filename.replace('.txt', ''), 
                school_conf_col] = school_conf
    
    return df


def add_detr_conf_scores(original_df, predictions_folder, school_conf_col = "conf_detr", image_id_col = "image_id"):
    """
    Get probability scores for school from DeTR model prediction text files and add them to DataFrame.
    Text files contains predictions in (class, x, y, w, h, conf) format.
    
    Args:
        original_df (pandas DataFrame): DeTR probability scores will be added to the DataFrame
        predictions_folder (str): folder containing DeTR model prediction text files
        school_conf_col (str): column name for DeTR probability scores in DataFrame
        image_id_col (str): column name for image ids in DataFrame
        
    Returns:
        DataFrame with DeTR model probability scores added in a separate column
    """
    
    df = original_df.copy()

    # parse DeTR prediction text files
    pred_text_list = os.listdir(predictions_folder)

    # loop through each prediction text file
    for filename in tqdm(pred_text_list):

        text_dir = os.path.join(predictions_folder, filename)
    
        # read prediction text file and get conf score
        with open(text_dir) as file:
            data = file.readlines()
            
            if len(data) == 0:
                continue
            
            conf_list = []

            for line in data:
                line = line.strip()
                conf = float(line.split(' ')[-1])
                conf_list.append(conf)

            # multiple bbox could be detected, take the max conf score
            school_conf = max(conf_list)  

        # add current image prediction conf score to DataFrame
        df.loc[ df[image_id_col] == filename.replace('.txt', ''), 
                school_conf_col] = school_conf

    return df



####################  For splitting outside/inside country admin boundary points  ####################

def add_outside_admin_boundary_column(df, admin_boundary_df, lat_col_name, lon_col_name, outside_boundary_col_name, crs = "EPSG:4326"):
    """
    Adds a separate column that flags points that are outside country admin boundary.
    Performs geopandas "inner" sjoin to split points that are inside and outside country admin boundary polygon.
    
    Args:
        df (pandas DataFrame): DataFrame containing lat / lon column
        admin_boundary_df (geopandas GeoDataFrame): country admin boundary polygon.
        lat_col_name (str): column name for latitude. e.g. "lat", "latitude"
        lon_col_name (str): column name for longitude. e.g. "lon", "longitude"
        outside_boundary_col_name (str): column name for points outside country admin boundary
        crs (str): original crs of the government school csv file. e.g. "EPSG:4326"
        
    Returns:
        df (pandas DataFrame): DataFrame with a new column (outside-country-boundary points flagged as "Yes")
    """
    
    new_df = df.copy()
    
    # if country admin boundary polygon crs is not target crs
    if admin_boundary_df.crs != crs:
        admin_boundary_df.to_crs(crs)
        
    # create a temporary DataFrame with geometry column
    temp_df = df[[lat_col_name, lon_col_name]].copy()

    # create geometry
    geometry = gpd.points_from_xy(temp_df[lon_col_name],
                                  temp_df[lat_col_name],
                                  crs = crs)

    # create GeoDataFrame by adding "geometry" column
    temp_gdf = gpd.GeoDataFrame(temp_df,
                                geometry = geometry)

    # intersect to create a DataFrame with only points inside country admin boundary
    inside_boundary_points_df = gpd.sjoin(temp_gdf,
                                          admin_boundary_df, 
                                          how='inner')

    # create a DataFrame with only points outside country admin boundary
    outside_boundary_points_df = temp_gdf[ ~temp_gdf.index.isin(inside_boundary_points_df.index) ]

    # indexes of points inside/outside country admin boundary
    outside_boundary_points_index = list(outside_boundary_points_df.index)
    inside_boundary_points_index  = list(inside_boundary_points_df.index)
    
    # add outside boundary column
    new_df.loc[new_df.index.isin(outside_boundary_points_index), 
               outside_boundary_col_name] = "Yes"
    
    return new_df



####################  For nearest neighbour operations  ####################

def create_nn_clusters(df, x_col_name, y_col_name, original_crs, nn_distance = 300, min_samples = 2):
    """
    Create clusters for input DataFrame based on nearest neighbour distance.
    Currently using DBSCAN.
    Returns GeoDataFrame with crs = "EPSG:3857".
    Reference - https://gis.stackexchange.com/questions/436238/find-multiple-nearby-points-in-a-dataframe-python
    
    Args:
        df (pandas.core.frame.DataFrame): DataFrame containing x,y coordinate values.
        x_col_name (str): name of x coordinates column in DataFrame. e.g. "longitude"
        y_col_name (str): name of y coordinates column in DataFrame. e.g. "latitude"
        original_crs (str): CRS of the input DataFrame for creating GeoDataFrame.e.g. "EPSG:4326"
        nn_distance (int): nearest neighbor distance (in meters). e.g. 300
        min_samples (int): minimum samples in a single cluster. e.g. 2
        
    Returns:
        gdf (geopandas.geodataframe.GeoDataFrame): a GeoDataFrame with NN cluster id column added 
    """

    cluster_id_col = 'cluster'
    
    # df = df.copy()

    # create a GeoDataFrame
    gdf = gpd.GeoDataFrame(df,
                           geometry = gpd.points_from_xy(df[x_col_name],
                                                         df[y_col_name]),
                           crs = original_crs)
    gdf = gdf.to_crs("EPSG:3857") # convert CRS
                                                  
    # calculate x and y columns
    gdf['x'] = gdf.geometry.x # create a x coordinate column (with units in meters)
    gdf['y'] = gdf.geometry.y

    # cluster
    # create a numpy array where each row is a coordinate pair
    # e.g. array([[ 694455.37762641, 7455507.93923842],
    #             [ 659868.57157602, 7504387.51909201], ...])
    coords = gdf[['x','y']].values

    # <nn_distance> is max distance between points
    # min cluster size = 2 points.
    # tweak the distance
    db = DBSCAN(eps = nn_distance, 
                min_samples = min_samples).fit(coords)
    
    cluster_labels = pd.Series(db.labels_).rename(cluster_id_col) # a series with all points cluster ids

    # concat cluster labels to the DataFrame
    gdf = pd.concat([gdf, cluster_labels], 
                    axis=1) 

    # drop x, y columns
    gdf.drop(['x', 'y'],
             axis = 1,
             inplace = True)
    
    return gdf
