import math
import traceback
import pandas as pd

class DataUtils:
    """
    An object to transform the data into an appropriate structure to be used for the genetic algorithm
    """
    def __init__(self, path: pd.DataFrame, metadata: pd.DataFrame):
        self.path = path
        self.metadata = metadata

        self.lat_long = None
        self.path_map = dict()
        self.cities = None

    def _clean_path_data(self):
        """
        Clean the map data
        :return: None
        """
        self.path.columns = [i.strip().lower() for i in self.path.columns]
        for col in self.path.columns:
            if self.path[col].dtype.str == 'object':
                self.path[col] = self.path[col].str.strip()

        grp_obj = self.path.groupby('source')
        for city, connx_df in grp_obj:
            connx = connx_df[["destination", "distance"]].apply(lambda x: tuple(x), axis=1).tolist()
            self.path_map[city] = connx

        self.cities = list(set(self.path.source.to_list() + self.path.destination.to_list()))

    def _clean_metadata(self) -> None:
        """
        Clean latitude longitude metadata and store it in a dict
        :return: None
        """
        self.metadata.columns = [i.strip().lower() for i in self.metadata.columns]
        for col in self.metadata.columns:
            if self.metadata[col].dtype.name == "object":
                self.metadata[col] = self.metadata[col].str.strip()
        self.lat_long = self.metadata.set_index("city").to_dict()

    def run(self) -> bool:
        try:
            self._clean_metadata()
            self._clean_path_data()
            return True
        except Exception as e:
            print(traceback.format_exception(e))
            return False
        

def haversine(coord1: tuple, coord2: tuple) -> float:
    """
    Haversine distance. Minimum distance on the earth's surface between any
    two points. This is calculated as:

    :param coord1: latitude & longitude of point 1
    :param coord2: latitude & longitude of point 2
    :return: shorted distance in KMs between two points on the surface of the earth
    """
    lat1, long1 = coord1
    lat2, long2 = coord2

    long1_r, lat1_r, long2_r, lat2_r = map(math.radians, [long1, lat1, long2, lat2])

    dlat = lat2_r - lat1_r
    dlong = long2_r - long1_r

    hav = math.sin(dlat / 2) ** 2 + (math.cos(lat1_r) * math.cos(lat2_r) * (math.sin(dlong / 2) ** 2))

    c = 2 * math.atan2(math.sqrt(hav), math.sqrt(1 - hav))
    earth_radius = 6731  # earth's raidus in KMs
    dist = earth_radius * c
    return dist


if __name__ == "__main__":
    pass