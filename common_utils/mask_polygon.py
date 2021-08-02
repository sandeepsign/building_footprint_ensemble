import numpy as np
from imantics import Mask
from sklearn.cluster import mean_shift
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection
import shapely.prepared
import shapely.vectorized
from alphashape import alphashape

from .numpy_utils import numpy_jsonify

DEFAULT_BANDWIDTH = 50      # Default bandwidth for mean shift
MEAN_SHIFT_ITERS  = 150     # Max number of iterations for mean shift

# A class for managing polygons that represents masks for image segmentation.
class MaskPolygon:
    def __init__(self, points):
        self.poly = Polygon(points)

    @property
    def points(self):
        all_points = np.array(self.poly.exterior.coords, dtype=int)
        #all_points = np.array(self.poly.exterior.simplify(200.0, preserve_topology=True).coords, dtype=int)
        return all_points[:-1] # last point is the same as first point

    @property
    def points_simple(self):
        # all_points = np.array(self.poly.exterior.coords, dtype=int)
        all_points = np.array(self.poly.exterior.simplify(10.0, preserve_topology=True).coords, dtype=int)
        return all_points[:-1] # last point is the same as first point

    # Returns the number of coordinates that define the polygon.
    @property
    def num_points(self):
        return len(self.points)

    # Returns the total area that the polygon covers.
    @property
    def area(self):
        return self.poly.area

    # Returns the total perimeter that the polygon covers.
    @property
    def perimeter(self):
        return self.poly.length

    # Returns the center of the polygon. Note that this could be outside the polygon itself
    @property
    def center(self):
        return np.average(self.points, axis=0).astype(int).tolist()

    # Returns a point that's pretty close to the center and guaranteed to be within the polygon
    @property
    def cheap_center(self):
        return [int(coord) for coord in self.poly.representative_point().coords[0]]

    # Approximate the inside centers via mean shift clustering
    # Afterwards, any inner centers near the polygon's center within
    # a certain distance are removed
    def inner_centers(self, remove_near_center_dist=0):
        self.fill_with_points()

        centers, _ = mean_shift(
            self.points,
            bandwidth=DEFAULT_BANDWIDTH,
            max_iter=MEAN_SHIFT_ITERS,
            bin_seeding=True
        )

        inner_centers = np.round(centers).astype(int)

        # Now let's remove any inner centers nearby the original center
        faraway_center_indices = np.linalg.norm(inner_centers - self.center, axis=1) > remove_near_center_dist
        inner_centers = inner_centers[faraway_center_indices]

        return inner_centers.tolist()

    # Returns the bounding box of the polygon in the following form: (minx, miny, maxx, maxy)
    @property
    def bbox(self):
        return [int(bnd) for bnd in self.poly.bounds]

    # Fills the polygon with points, since it's originally defined by the polygon's exterior coords
    # This is particularly useful when creating a concave hull or finding inner centers of said polygon
    # The points to area ratio is analog to how much of a polygon is filled with points, with a range from 0 to 1
    # The higher the ratio, the more points are potentially added into said polygon
    def fill_with_points(self, points_to_area_ratio=0.2):
        (minx, miny, maxx, maxy) = self.poly.bounds
        potential_points = []
        prepped_poly = shapely.prepared.prep(self.poly)

        while (self.num_points + len(potential_points)) / self.area < points_to_area_ratio:
            needed_num_points = int((self.num_points + len(potential_points)) * points_to_area_ratio)
            ptsX = np.random.uniform(minx, maxx, needed_num_points).astype(int)
            ptsY = np.random.uniform(miny, maxy, needed_num_points).astype(int)
            pts = np.column_stack((ptsX, ptsY))

            indicies = shapely.vectorized.contains(prepped_poly, ptsX, ptsY)
            potential_points += [*pts[indicies]]

        potential_points += self.poly.exterior.coords
        self.poly = Polygon(potential_points)

    # Returns the concave hull of the polygon, based on alpha shapes
    # If alpha isn't specified, an alpha value will be optimized for one
    def concave_hull(self, alpha=None):
        self.fill_with_points()

        optimized = alphashape(self.points, alpha)
        if isinstance(optimized, MultiPolygon) or isinstance(optimized, GeometryCollection):
            optimized = list(optimized.geoms)
        else:
            optimized = [optimized]

        return optimized

    # Simplifies the polygon within a distance tolerance.
    # 'preserve_topology' should be true, or else the algorithm may produce
    # self-intersecting or otherwise invalid geometries
    def simplify(self, dist_tolerance, preserve_topology=True):
        self.poly = self.poly.simplify(dist_tolerance, preserve_topology)

    def get_simple_json(self):
        return {
            'num_points': self.num_points,
            'perimeter': self.perimeter,
            'area': self.area,
            'center': self.center
        }

    def get_full_json(self):
        return {
            'num_points': self.num_points,
            'perimeter': self.perimeter,
            'area': self.area,
            'center': self.center,
            'points': numpy_jsonify(self.points),
        }
