import cv2
import numpy as np

# A class for managing tile splitting and recombining for smoother prediction masks
class ImageTileManager:
    def __init__(self, tile_size, tile_stride):
        self.tile_size = tile_size
        self.tile_stride = tile_stride

    # Splits a full sized image into square tiles
    def get_tiles(self, img, nested=False):
        (num_rows, num_cols) = img.shape[:2]
        tile_sz = self.tile_size
        tile_sk = self.tile_stride
        img_rows = []

        for r in range(0, num_rows, tile_sk):
            imgs_per_row = []
            if r + tile_sz > num_rows:
                break

            for c in range(0, num_cols, tile_sk):
                if c + tile_sz > num_cols:
                    break

                sub_square = img[r:r+tile_sz, c:c+tile_sz]
                imgs_per_row.append(sub_square)

            img_rows.append(imgs_per_row)

        img_tiles = np.array(img_rows)

        return img_tiles if nested else self.unravel_tiles(img_tiles)

    # Recombines an array or grid of image tiles into a full sized image
    # If averaging overlap is true, then the image tile values are smoothed out if the
    # stride length is less than the length of the tile.
    def recombine_tiles(self, img_tiles, orig_img_shape, tile_dim=None, avg_overlap=False):
        tile_sz = self.tile_size
        tile_sk = self.tile_stride

        if self.is_unraveled(img_tiles):
            img_tiles = self.gridify_tiles(img_tiles, tile_dim)

        if not avg_overlap:
            recombined_img = np.zeros(orig_img_shape, dtype=np.uint8)

            for (r, img_row) in enumerate(img_tiles):
                for (c, sub_img) in enumerate(img_row):
                    recombined_img[r*tile_sk:r*tile_sk+tile_sz, c*tile_sk:c*tile_sk+tile_sz] = np.squeeze(sub_img)
        else:
            recombined_img = np.zeros(orig_img_shape, dtype=np.uint32)
            averaging_mask = np.zeros(orig_img_shape, dtype=np.uint32)
            max_val = np.max(img_tiles)

            for (r, img_row) in enumerate(img_tiles):
                for (c, sub_img) in enumerate(img_row):
                    recombined_img[r*tile_sk:r*tile_sk+tile_sz, c*tile_sk:c*tile_sk+tile_sz] += np.squeeze(sub_img)
                    averaging_mask[r*tile_sk:r*tile_sk+tile_sz, c*tile_sk:c*tile_sk+tile_sz] += max_val

            recombined_img = (recombined_img / averaging_mask * max_val).astype(np.uint8)

        return recombined_img

    # Checks if an array-like object is a grid (n x m) of image tiles that fit the tile manager's parameters
    def is_valid_grid(self, img_tiles):
        tiles_shape = img_tiles.shape
        if len(tiles_shape) == 4:
            return tiles_shape[1] == self.tile_size and tiles_shape[2] == self.tile_size
        elif len(tiles_shape) == 5:
            return tiles_shape[2] == self.tile_size and tiles_shape[3] == self.tile_size
        else:
            return False

    # Checks if the image tiles is arranged like a grid (n x m)
    def is_gridified(self, img_tiles):
        if not self.is_valid_grid(img_tiles):
            raise Exception(f'Not a valid grid of image tiles')
        return len(img_tiles.shape) == 5

    # Checks if the image tiles is arranged like a line of tiles of length (n x m)
    def is_unraveled(self, img_tiles):
        if not self.is_valid_grid(img_tiles):
            raise Exception(f'Not a valid grid of image tiles')
        return len(img_tiles.shape) == 4

    # A helper method to unravel a grid of image tiles into a single line (useful prior to model inferencing)
    @classmethod
    def unravel_tiles(self, img_tiles):
        len_dims = len(img_tiles.shape)
        if len_dims == 5:
            return np.reshape(img_tiles, (-1, *img_tiles.shape[2:]))
        else:
            raise Exception(f'Invalid dimension size: Should have 5 dimensions only, but got {len_dims} dimensions')

    # A helper method to turn a line of image tiles into a grid (useful prior to tile recombining)
    @classmethod
    def gridify_tiles(self, img_tiles, tile_dim=None):
        len_dims = len(img_tiles.shape)
        if len_dims == 4:
            if tile_dim is None:
                num_tiles = img_tiles.shape[0]
                tile_len = int(np.round(num_tiles**.5))
                tile_dim = (tile_len, tile_len)
            return np.reshape(img_tiles, (tile_dim[0], tile_dim[1], *img_tiles.shape[1:]))
        else:
            raise Exception(f'Invalid dimension size: Should have 4 dimensions only, but got {len_dims} dimensions')
