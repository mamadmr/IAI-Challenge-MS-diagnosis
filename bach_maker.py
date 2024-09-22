import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import os 
import json 

class BatchMaker:
    def __init__(self) -> None:
        pass

    def get_the_paths(self, default_path: str) -> dict[int, dict[str, str]]:
        """
        Scans the given directory for subdirectories named with digits, and within each, finds files
        containing 'flair' and 'lesion' in their filenames. It constructs a dictionary mapping each
        folder number to the paths of these files.

        Args:
            default_path (str): The root directory containing the subfolders to scan.

        Returns:
            Dict[int, Dict[str, str]]: A dictionary where each key is an integer representing a folder
            name, and each value is another dictionary with keys 'mri' and 'lesion' mapping to their
            respective file paths.

        Raises:
            FileNotFoundError: If the provided `default_path` does not exist or is not a directory.
            ValueError: If a folder does not contain both a 'flair' and a 'lesion' file.

        Example:
            Given a directory structure:
                default_path/
                    1/
                        subject1_flair.nii
                        subject1_lesion.nii
                    2/
                        subject2_flair.nii
                        subject2_lesion.nii

            Calling `get_the_paths(default_path)` would return:
                {
                    1: {
                        'mri': 'default_path/1/subject1_flair.nii',
                        'lesion': 'default_path/1/subject1_lesion.nii'
                    },
                    2: {
                        'mri': 'default_path/2/subject2_flair.nii',
                        'lesion': 'default_path/2/subject2_lesion.nii'
                    }
                }
        """

        if not os.path.isdir(default_path):
            raise FileNotFoundError(f"The directory '{default_path}' does not exist.")
        
        paths = dict()
        folders = os.listdir(default_path)

        for folder in folders:
            if not folder.isdigit():
                continue

            files = os.listdir(default_path+'/'+folder)

            for file in files:
                if 'flair' in file:
                    mri_path = default_path+'/'+folder+'/'+file
                if 'lesion' in file:
                    lesion_path = default_path+'/'+folder+'/'+file

            paths[int(folder)] = {'mri': mri_path, 'lesion': lesion_path}
        
        return paths
    
    def load_image(self, path) -> np.memmap:
        """
            Loads a medical image file using nibabel and returns the image data as a NumPy memmap array.

            Args:
                path (str): The file path to the medical image.

            Returns:
                np.memmap: The image data array as a memory-mapped array.

            Raises:
                FileNotFoundError: If the file at the given path does not exist.
                nibabel.filebasedimages.ImageFileError: If the file cannot be loaded as an image.
                Exception: For any other exceptions raised during image loading.

            Example:
                image_data = self.load_image('/path/to/image.nii')
                print(image_data.shape)
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The file '{path}' does not exist.")
        
        image = nib.load(path)
        return image.get_fdata()
    
    def BFS(self, lesion) -> list[dict]:
        """
            Performs a Breadth-First Search (BFS) on a 3D numpy array to identify and locate connected lesions.

            Parameters:
                lesion (np.ndarray): A 3D numpy array where non-zero values represent lesion voxels.

            Returns:
                List[dict]: A list of dictionaries, each representing a lesion (connected component) in the volume.
                    Each dictionary contains:
                        - 'start' (List[int]): The starting coordinates [x1, y1, z1] of the bounding box enclosing the lesion.
                        - 'end' (List[int]): The ending coordinates [x2, y2, z2] of the bounding box enclosing the lesion.
                        - 'lesion_value' (int or float): The value representing the lesion type.
                        - 'dimensions' (List[int]): The dimensions [dx, dy, dz] of the bounding box.

            Notes:
                - The function explores 26-connected neighborhoods (including diagonals) for connectivity.
                - Multiple lesions (connected components) of different types (non-zero values) can be present in the array.

        """
        
        output = []
        mark = np.zeros_like(lesion, dtype=bool)

        # find the starting points of the lesion
        for i in range(lesion.shape[0]):
            for j in range(lesion.shape[1]):
                for k in range(lesion.shape[2]):
                    if lesion[i, j, k] == 0 or mark[i, j, k] == 1:
                        continue

                    # start BFS
                    queue = [(i, j, k)]
                    lesion_type = lesion[i, j, k]
                    mark[i, j, k] = 1
                    points = [(i, j, k)]

                    while queue:
                        x, y, z = queue.pop(0)

                        # check the neighbours
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    if dx == 0 and dy == 0 and dz == 0:
                                        continue

                                    if x + dx < 0 or x + dx >= lesion.shape[0] or y + dy < 0 or y + dy >= lesion.shape[1] or z + dz < 0 or z + dz >= lesion.shape[2]:
                                        continue

                                    if lesion[x + dx, y + dy, z + dz] == lesion_type and mark[x + dx, y + dy, z + dz] == 0:
                                        mark[x + dx, y + dy, z + dz] = 1
                                        points.append((x + dx, y + dy, z + dz))
                                        queue.append((x + dx, y + dy, z + dz))
                    
                    # find the regtangle that contains the lesion 
                    points = np.array(points).T 
                    start = np.min(points, axis=1)
                    end = np.max(points, axis=1)
                    dimensions = end - start + 1
                    output.append({'start': list([int(temp) for temp in start]),
                                    'end': list([int(temp) for temp in end]), 'type': lesion_type, 
                                    'dimensions': list([int(temp) for temp in dimensions])})
    
        return output
         
    def find_lesion_cordiantes(self, lesion):
        '''
            this method use BFS to find the coordinates of a regtangles that contains the lesion
        '''
        regtangles = self.BFS(lesion)

        return regtangles

    def load_regtangles(self, file_path: str, data_path: str=None) -> dict:
        """
        Loads lesion bounding rectangles from a JSON file if it exists; otherwise, generates the rectangles
        by processing lesion images and saves them to the JSON file for future use.

        Parameters:
            file_path (str): The file path to the JSON file containing precomputed lesion rectangles.
            data_path (str): The directory path where lesion data images are stored.

        Returns:
            dict: A dictionary containing lesion rectangles for each sample, keyed by sample identifier.

        Notes:
            - If the JSON file does not exist at `file_path`, the function generates the lesion
            rectangles by processing the lesion images found in `data_path` and saves the results
            to the JSON file specified by `file_path`.
            - Progress is printed to the console as a percentage.
            - The method relies on the following methods being defined in the class:
                - `get_the_paths(data_path: str) -> dict`
                - `load_image(path: str) -> np.ndarray`
                - `find_lesion_coordinates(lesion: np.ndarray) -> list`

        Example:
            rectangles = self.load_rectangles('lesion_rectangles.json', '/path/to/data')
            print(rectangles['1'])  # Outputs the lesion rectangles for sample 1.

        Raises:
            FileNotFoundError: If `data_path` does not exist.
            Exception: If any of the required methods are not implemented.
        """

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                output = json.load(file)
            return output
        print('file not found\ngenerating the regtangles file\n')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The data path '{data_path}' does not exist.")

        paths = self.get_the_paths(data_path)
        output = dict()
        for i in sorted(paths.keys()):
            sample_lesion = paths[i]['lesion']
            lesion = self.load_image(sample_lesion)
            regtangles = self.find_lesion_cordiantes(lesion)
            output[str(i)] = regtangles
            print('\033[F', end='')
            print(f'{(i/len(paths))*100:.2f}%')

        print("generating the regtangles file is done\n")

        with open(file_path, 'w') as file:
            json.dump(output, file)       

        return output

    def lesion_color_map(self, lesion_slice, to_display):
        colors = np.zeros((*lesion_slice.shape, 4))  # RGBA color space
        if to_display[0] == 1:
            colors[lesion_slice == 1] = [1, 0, 0, 0.5]  # Red with transparency
        
        if to_display[1] == 1:
            colors[lesion_slice == 2] = [0, 1, 0, 0.5]  # Green with transparency
        
        if to_display[2] == 1:
            colors[lesion_slice == 3] = [0, 0, 1, 0.5]  # Blue with transparency
        
        if to_display[3] == 1:
            colors[lesion_slice == 4] = [1, 1, 0, 0.5]  # Yellow with transparency
        return colors

    def display_3d_plotly(self, lesion_data, regtangles, to_display):
        x, y, z = np.indices(lesion_data.shape)
        
        # Flatten the data for plotting
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        lesion_data_flat = lesion_data.flatten()

        # Create a 3D scatter plot of the lesions
        fig = go.Figure()

        # Loop through the lesion categories and plot them with their corresponding colors
        for category in np.unique(lesion_data):
            if category == 0:  # Skip background
                continue
            if to_display[int(category)-1] == 0:
                continue

            mask = lesion_data_flat == category
            color = self.lesion_color_map(np.array([category]), to_display)[0][:3]  # Get the RGB part of the color
            
            fig.add_trace(go.Scatter3d(
                x=x[mask],
                y=y[mask],
                z=z[mask],
                mode='markers',
                marker=dict(
                    size=1,
                    color=f'rgb({color[0] * 255}, {color[1] * 255}, {color[2] * 255})',
                    opacity=0.5
                ),
                name=f'Lesion category {int(category)}'
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            title=f'3D Lesion Data',
            autosize=True
        )
        
        # Add rectangles (bounding boxes) to the plot using lines
        for rectangle in regtangles:
            type = rectangle['type']

            if to_display[int(type)-1] == 0:
                continue

            start = rectangle['start']
            end = rectangle['end']
            
            # Create the 8 corners of the cuboid
            corners = [
                (start[0], start[1], start[2]),  # (x1, y1, z1)
                (end[0], start[1], start[2]),    # (x2, y1, z1)
                (end[0], end[1], start[2]),      # (x2, y2, z1)
                (start[0], end[1], start[2]),    # (x1, y2, z1)
                (start[0], start[1], end[2]),    # (x1, y1, z2)
                (end[0], start[1], end[2]),      # (x2, y1, z2)
                (end[0], end[1], end[2]),        # (x2, y2, z2)
                (start[0], end[1], end[2]),      # (x1, y2, z2)
            ]
            
            # Define the edges connecting the corners
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom rectangle
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top rectangle
                (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical lines connecting top and bottom
            ]
            
            # Add the lines for the bounding box
            for edge in edges:
                fig.add_trace(go.Scatter3d(
                    x=[corners[edge[0]][0], corners[edge[1]][0]],
                    y=[corners[edge[0]][1], corners[edge[1]][1]],
                    z=[corners[edge[0]][2], corners[edge[1]][2]],
                    mode='lines',
                    line=dict(color='black', width=3),
                    name='Bounding Box'
                ))

        fig.show()

    def generate_random_patch_with_lesion(self, lesion_data, regtangles, patch_size: tuple):
        x, y, z = patch_size
        # Select a random regtangle
        regtangle = regtangles[np.random.randint(0, len(regtangles))]
        start = np.array(regtangle['start'])
        end = np.array(regtangle['end'])
        middle = (start + end) // 2

        # randomize the patch center
        center = np.array([np.random.randint(middle[0] - x // 2, middle[0] + x // 2),
                           np.random.randint(middle[1] - y // 2, middle[1] + y // 2),
                           np.random.randint(middle[2] - z // 2, middle[2] + z // 2)])
        
        start = center - np.array([x // 2, y // 2, z // 2])
        start = np.maximum(start, 0)
        start = np.minimum(start, np.array(lesion_data.shape) - np.array([x, y, z]))

        return lesion_data[start[0]:start[0] + x, start[1]:start[1] + y, start[2]:start[2] + z], start 
    
    def generate_random_patch_without_lesion(self, lesion_data, patch_size: tuple, threshold: int):
        x, y, z = patch_size
        
        # find a patch that the number of the pixels that are in the lesion is less than the threshold
        while True:
            start = np.array([np.random.randint(0, lesion_data.shape[0] - x),
                            np.random.randint(0, lesion_data.shape[1] - y),
                            np.random.randint(0, lesion_data.shape[2] - z)])
        
            patch = lesion_data[start[0]:start[0] + x, start[1]:start[1] + y, start[2]:start[2] + z]

            temp_patch = patch > 0 
            if np.sum(temp_patch) < threshold:
                return patch, start
        
    def generate_batch(self, regtangle_path, data_path, patch_size, batch_size, positive_ratio, threshold):
        regtangles = self.load_regtangles(regtangle_path, data_path)
        paths = self.get_the_paths(data_path)

        batch_mri = []
        batch_lesion = []
        start_points = []

        for sample_num in range(batch_size):
            # choose a random image
            i = np.random.randint(1, len(paths)+1)

            if np.random.rand() < positive_ratio:
                _, start = self.generate_random_patch_with_lesion(self.load_image(paths[i]['lesion']), regtangles[str(i)], patch_size)
            
            else:
                _, start = self.generate_random_patch_without_lesion(self.load_image(paths[i]['lesion']), patch_size, threshold)
            
            mri_image = self.load_image(paths[i]['mri'])
            batch_mri.append(mri_image[start[0]:start[0] + patch_size[0], start[1]:start[1] + patch_size[1], start[2]:start[2] + patch_size[2]])

            lesion_image = self.load_image(paths[i]['lesion'])
            batch_lesion.append(lesion_image[start[0]:start[0] + patch_size[0], start[1]:start[1] + patch_size[1], start[2]:start[2] + patch_size[2]])

            start_points.append(start)

            print('\033[F', end='')
            print(f'{(sample_num/batch_size)*100:.2f}%')

        return np.array(batch_mri), np.array(batch_lesion), np.array(start_points)


if __name__ == '__main__':
    #num = 37
    batch_maker = BatchMaker()
    #reg = batch_maker.load_regtangles('regtangles.json', 'images')
    x = batch_maker.generate_batch('regtangles.json', 'images', (64, 64, 64), 30, 0.8) 

    print(x[0].shape)
    print(x[1].shape)

    #z = batch_maker.load_image(batch_maker.get_the_paths('images')[1]['lesion'])
    #print(type(z))
    #paths = batch_maker.get_the_paths('images')
    #lesion = batch_maker.load_image(paths[num]['lesion'])

    #batch_maker.display_3d_plotly(lesion, reg[str(num)], [1, 1, 1, 1])
    #patch, start = batch_maker.generate_random_patch_with_lesion(lesion, reg[str(num)], (64, 64, 64))

    #print(patch.shape, start)

    #batch_maker.display_3d_plotly(patch, [], [1, 1, 1, 1])

    #patch, start = batch_maker.generate_random_patch_without_lesion(lesion, (64, 64, 64), 20)

    #print(patch.shape, start)

    #batch_maker.display_3d_plotly(patch, [], [1, 1, 1, 1])


    #x = batch_maker.find_lesion_cordiantes(batch_maker.load_image(batch_maker.get_the_paths('images')[1]['lesion']))

    #print(x[0])
    '''
    for i in range(1, len(paths)+1):
        sample_lesion = paths[i]['lesion']
        lesion = batch_maker.load_image(sample_lesion)
        regtangles = batch_maker.find_lesion_cordiantes(lesion)
        x, y, z = 0, 0, 0
        for regtangle in regtangles:
            x = max(x, regtangle['dimensions'][0])
            y = max(y, regtangle['dimensions'][1])
            z = max(z, regtangle['dimensions'][2])
    
        print(i, ":", x, y, z)
    '''
    #sample_lesion = paths[13]['lesion']
    #lesion = batch_maker.load_image(sample_lesion)
    #regtangles = batch_maker.find_lesion_cordiantes(lesion)
    
    #print(regtangles)
    #print(len(regtangles))
    #batch_maker.display_3d_plotly(lesion, regtangles, [1, 0, 0, 0])