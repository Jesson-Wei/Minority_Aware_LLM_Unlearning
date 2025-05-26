import os
from dataclasses import dataclass, field

@dataclass
class OutdirArgs:
    CONFIG_KEY = "outdir_args"

    root: str = field(default=f'{os.getcwd()}/experiments', metadata={
        "help": "Root folder where to put the experiments"
    })

    name: str = field(default="experiment", metadata={
        "help": "Name of each experiment folder"
    })

    folder_number: str = field(default=None, metadata={
        "help": "Suffix of each folder (e.g., '00001')"
    })

    custom_output_dir: str = field(default=None, metadata={
        "help": "Specify a custom output directory directly (overrides automatic folder creation)."
    })

    def create_folder_name(self):
        """ Gets a unique folder name and creates it if it does not exist """
        # If custom_output_dir is specified in the YAML, it will be used preferentially
        if self.custom_output_dir is not None:
            os.makedirs(self.custom_output_dir, exist_ok=True)
            return os.path.abspath(self.custom_output_dir)
        
        # Otherwise, create the folder using the automatically generated method
        folder_name = self._get_folder_path()
        os.makedirs(folder_name, exist_ok=True)
        return os.path.abspath(folder_name)

    def _get_folder_path(self):
        """ Get an unused folder name in the root directory. """
        if self.folder_number is None:
            os.makedirs(self.root, exist_ok=True)
            numbers = [0]
            for folder in [x for x in os.listdir(self.root) if self.name in x]:
                numbers.append(int(folder.split(self.name + "_")[1]))
            self.folder_number = str(int(max(numbers) + 1)).zfill(5)
        return os.path.join(self.root, f'{self.name}_{self.folder_number}')
