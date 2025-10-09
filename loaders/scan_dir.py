import os

class Experiment:
    def __init__(self, name, tiff_path, nii_paths, parent_dir):
        self.name = name
        self.tiff = tiff_path
        self.niis = nii_paths
        self.parent_dir = parent_dir

    def __repr__(self):
        return f"<Experiment name={self.name} tiff={self.tiff} niis={len(self.niis)}>"


class ListExperiments:
    def __init__(self, root):
        self.root = root
        self.experiments = []

    def list_files(self):
        self.experiments = []

        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [
                d for d in dirnames if not d.startswith(".") and d != "__MACOSX"
            ]

            tiffs = [f for f in filenames if f.lower().endswith((".tif", ".tiff"))]
            if not tiffs:
                continue

            niis = [f for f in filenames if f.lower().endswith(".nii")]
            if not niis:
                continue

            parent_dir = os.path.basename(dirpath)
            experiment = Experiment(
                name=parent_dir,
                tiff_path=os.path.join(dirpath, tiffs[0]),
                nii_paths=[os.path.join(dirpath, n) for n in niis],
                parent_dir=parent_dir
            )
            self.experiments.append(experiment)

        print(f"Found {len(self.experiments)} experiments under {self.root}")
        return self.experiments

    def display_experiments(self):
        if not self.experiments:
            print("No experiments to display. Did you run list_files()?")
            return

        for exp in self.experiments:
            print(f"\n{exp.name}")
            print(f"  Parent Dir: {exp.parent_dir}")
            print(f"  TIFF: {exp.tiff}")
            for nii in exp.niis:
                print(f"   - {nii}")

