import pandas as pd
import re

class ReadMetadataReader:
    def __init__(self, file_path, sheet_name=None, dearry_map_file= None):
        """
        Initialize the ReadMetadata class with the path to the Excel file and optional sheet name.
        
        :param file_path: Path to the Excel file containing metadata.
        :param sheet_name: Name of the sheet to read from the Excel file. If None, the first sheet is used.
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.dearry_map_file = dearry_map_file
        self._init()

    def _init(self):
        """
        Read the Excel file and store the metadata in a dictionary.
        
        :return: Dictionary with slide names as keys and corresponding core labels as values.
        """
        if self.file_path is not None:
            df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            self.metadata = {}
            
            # Assuming the first row contains the slide names and the first column contains the core numbers
            slide_names = df.columns[1:-2]
            for slide in slide_names:
                slide = str(slide)  # Ensure slide name is a string
                self.metadata[slide] = {}
                for index, row in df.iterrows():
                    core_number = str(row.iloc[0])  # Ensure core number is a string
                    if slide in row:
                        self.metadata[slide][core_number] = row[slide]
                    else:
                        self.metadata[slide][core_number] = row[int(slide)]

        if self.dearry_map_file is not None:
            self.mapping = {}
            for line in open(self.dearry_map_file).readlines():
                label, core_number = line[:-1].split(',')
                self.mapping[label] = core_number
            print(self.mapping) 

    def get_dearray_mapping(self):
        if hasattr(self, 'dearry_map_file'):
            return self.mapping
        else:
            return None
    
    def _extract_numerical_part(self, slide_name):
        """
        Extracts and returns the numerical part of the slide name.

        Args:
            slide_name (str): The original slide name.

        Returns:
            str: The numerical part of the slide name.
        """
        numerical_parts = re.findall(r'\d+', slide_name)
        return ''.join(numerical_parts)

    def get_metadata(self, slide_name, core_number):
        """
        Retrieve metadata for a given slide name and core number.
        
        :param slide_name: The slide name for which metadata is to be retrieved.
        :param core_number: The core number for which metadata is to be retrieved.
        :return: Metadata for the specified slide name and core number.
        """
        return self.metadata.get(self._extract_numerical_part(slide_name), {}).get(core_number, None)

    def check_slide_exists(self, slide_name):
        """
        Check if metadata is available for a given slide name.
        
        :param slide_name: The slide name to be checked.
        :return: True if metadata is available for the specified slide name, False otherwise.
        """
        if hasattr(self, 'metadata'):
            return self._extract_numerical_part(slide_name) in self.metadata
        else:
            return False
    
    def get_number_of_cores(self):
        """
        Get the number of cores for which metadata is available.
        
        :return: Number of cores.
        """
        if hasattr(self, 'metadata'):
            return len(next(iter(self.metadata.values())))
        else:
            return -1
    
    def get_metadata_string(self, slide_name):
        """
        Retrieve and format metadata for a given slide name.
        
        :param slide_name: The slide name for which metadata is to be retrieved.
        :return: Formatted metadata string for the specified slide name.
        """
        if hasattr(self, 'metadata'):

            numerical_part = self._extract_numerical_part(slide_name)
            slide_metadata = self.metadata.get(numerical_part, {})

            if not slide_metadata:
                return "No metadata available for the given slide name."

            metadata_strings = []
            cores = list(slide_metadata.items())

            for i in range(0, len(cores), 2):
                core1 = cores[i]
                core2 = cores[i + 1] if i + 1 < len(cores) else None

                if core2:
                    metadata_strings.append(f"Core {core1[0]}: {core1[1]}, Core {core2[0]}: {core2[1]}")
                else:
                    metadata_strings.append(f"Core {core1[0]}: {core1[1]}")

            return "\n".join(metadata_strings)
        else:
            return ''


# Example usage:
# metadata_reader = ReadMetadataReader('CPQA.xlsx', 'BRAFV600E assessments')
# slide_metadata = metadata_reader.get_metadata('240', '1')
# print(metadata_reader.get_number_of_cores())    
# print(slide_metadata)