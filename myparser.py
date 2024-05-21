import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract areas from WSI slides based on the provided HistoQC masks")
    parser.add_argument("--slides_path", type=str, help="Path to the WSI slides location", required=True)
    parser.add_argument("--masks_path", type=str, help="Path to the HistoQC masks location", required=False)
    parser.add_argument("--annotations_path", type=str, help="Path to the slides' annotation.", required=False)
    parser.add_argument("--output_path", type=str, help="Path to the output location", required=True)
    parser.add_argument("--slide_down_sample_rate", type=int, help="the rate of down sampling the extracted regions", default=5)
    parser.add_argument("--overlay_down_sample_rate", type=int, help="the rate of down sampling to generated overlay", default=5)

    # Add deepliif arguments
    parser.add_argument("--deepliif", action='store_true', help="Run deepliif")
    parser.add_argument("--model_dir", type=str, help="Path to the unserilized model directory")
    parser.add_argument("--tile_size", type=int, help="Size of the tiles to be processed", default=256)
    parser.add_argument("--post_processing", type=bool, help="Run postprocessing alogrithem on the results", default=True)

    # Add mask generation arguments
    parser.add_argument("--mask_generator", action='store_true', help="Run mask generator")
    parser.add_argument("--model_path", type=str, help="Path to the segment anything model checkpoint")

    return parser.parse_args()

