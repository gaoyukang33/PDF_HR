# GMR to MimicKit Motion Data Conversion Guide

This guide demonstrates how to convert motion data from GMR format to MimicKit format and verify the contents of both input and output files.

**Prerequisites:** 

- All commands should be run from the root directory of the repository (`MimicKit/`).


---

## Convert GMR to MimicKit Format

Use the conversion script to transform your GMR motion data into MimicKit-compatible format.

**Command:**

```bash
python tools/data_format_conversion/gmr_to_mimickit.py --input_file {input_file_path} --output_file {output_file_path}
```

---

## Note:

The output motion file should be placed into `data/motions/{your dir}` for later use and it can be viewed using `view_motion_env`. This is also a good sanity test to see if your motion is converted correctly with `MimicKit`.
