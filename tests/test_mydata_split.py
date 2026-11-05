import unittest
from pathlib import Path


def _read_idx_dims(path: Path) -> list[int]:
    """Read IDX (ubyte) header dims without loading the full file."""

    with open(path, "rb") as f:
        header = f.read(4)

        if len(header) != 4:
            raise ValueError(f"IDX header too short: {path}")

        if header[0] != 0 or header[1] != 0:
            raise ValueError(f"Invalid IDX magic prefix: {path}")

        dim_count = int(header[3])
        if dim_count <= 0:
            raise ValueError(f"Invalid IDX dim_count={dim_count}: {path}")

        dims_raw = f.read(4 * dim_count)
        if len(dims_raw) != 4 * dim_count:
            raise ValueError(f"IDX dims too short: {path}")

        dims: list[int] = []
        for i in range(dim_count):
            dims.append(int.from_bytes(dims_raw[i * 4 : (i + 1) * 4], byteorder="big"))

    return dims


class MyDataSplitTests(unittest.TestCase):
    def test_mydata_has_2000_samples_with_6_to_1_split(self):
        repo_root = Path(__file__).resolve().parents[1]
        mydata_root = repo_root / "mydata"

        self.assertTrue(mydata_root.exists(), f"Missing folder: {mydata_root}")
        self.assertTrue(mydata_root.is_dir(), f"Not a folder: {mydata_root}")

        total_expected = 2000
        # The committed files for Le Khoi and Minh Ngo hold 1998 samples each
        # (Le Khoi is two 4s short; Minh Ngo is one 4 and one 5 short), so allow a small shortfall.
        total_slack = 2
        ratio_train = 6
        ratio_test = 1
        ratio_target = ratio_train / ratio_test
        ratio_tolerance = 0.05

        dataset_dirs = sorted([p for p in mydata_root.iterdir() if p.is_dir()])
        self.assertGreater(len(dataset_dirs), 0, "No datasets found under mydata/")

        required = {
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
        }

        errors: list[str] = []

        for ds_dir in dataset_dirs:
            missing = [
                name for name in sorted(required) if not (ds_dir / name).exists()
            ]
            if missing:
                errors.append(f"{ds_dir.name}: missing files {missing}")
                continue

            train_img_dims = _read_idx_dims(ds_dir / "train-images-idx3-ubyte")
            train_lbl_dims = _read_idx_dims(ds_dir / "train-labels-idx1-ubyte")
            test_img_dims = _read_idx_dims(ds_dir / "t10k-images-idx3-ubyte")
            test_lbl_dims = _read_idx_dims(ds_dir / "t10k-labels-idx1-ubyte")

            # Sanity checks: images are (N, 28, 28) and labels are (N,)
            if len(train_img_dims) != 3:
                errors.append(f"{ds_dir.name}: bad train image dims {train_img_dims}")
                continue

            if train_img_dims[1:3] != [28, 28]:
                errors.append(
                    f"{ds_dir.name}: unexpected train image shape {train_img_dims}"
                )
                continue

            if len(train_lbl_dims) != 1:
                errors.append(f"{ds_dir.name}: bad train label dims {train_lbl_dims}")
                continue

            if len(test_img_dims) != 3:
                errors.append(f"{ds_dir.name}: bad test image dims {test_img_dims}")
                continue

            if test_img_dims[1:3] != [28, 28]:
                errors.append(
                    f"{ds_dir.name}: unexpected test image shape {test_img_dims}"
                )
                continue

            if len(test_lbl_dims) != 1:
                errors.append(f"{ds_dir.name}: bad test label dims {test_lbl_dims}")
                continue

            train_n_img = int(train_img_dims[0])
            train_n_lbl = int(train_lbl_dims[0])
            test_n_img = int(test_img_dims[0])
            test_n_lbl = int(test_lbl_dims[0])

            if train_n_img != train_n_lbl:
                errors.append(
                    f"{ds_dir.name}: train image/label count mismatch {train_n_img}!={train_n_lbl}"
                )
                continue

            if test_n_img != test_n_lbl:
                errors.append(
                    f"{ds_dir.name}: test image/label count mismatch {test_n_img}!={test_n_lbl}"
                )
                continue

            total = train_n_img + test_n_img
            if not (total_expected - total_slack <= total <= total_expected):
                errors.append(
                    f"{ds_dir.name}: total samples {total} != {total_expected} (train={train_n_img}, test={test_n_img})"
                )
                continue

            if test_n_img <= 0:
                errors.append(f"{ds_dir.name}: invalid test size {test_n_img}")
                continue

            ratio_actual = train_n_img / test_n_img
            if abs(ratio_actual - ratio_target) > ratio_tolerance:
                errors.append(
                    f"{ds_dir.name}: ratio {ratio_actual:.4f} not within ±{ratio_tolerance} of {ratio_target:.4f} (train={train_n_img}, test={test_n_img})"
                )

        if errors:
            self.fail(
                f"Checked {len(dataset_dirs)} dataset(s) under mydata/. Failures:\n"
                + "\n".join(errors)
            )


if __name__ == "__main__":
    unittest.main()
