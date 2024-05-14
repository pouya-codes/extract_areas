import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract areas from WSI slides based on the provided HistoQC masks")
    parser.add_argument("--slides_path", type=str, help="Path to the WSI slides location", required=True)
    parser.add_argument("--masks_path", type=str, help="Path to the HistoQC masks location", required=False)
    parser.add_argument("--annotations_path", type=str, help="Path to the slides' annotation.", required=False)
    parser.add_argument("--output_path", type=str, help="Path to the output location", required=True)
    parser.add_argument("--slide_down_sample_rate", type=int, help="the rate of down sampling the extracted regions", default=5)
    parser.add_argument("--overlay_down_sample_rate", type=int, help="the rate of down sampling to generated overlay", default=5)
    

    # Add a subparser for deepliif arguments
    subparsers = parser.add_subparsers(dest='deepliif')
    deepliif_parser = subparsers.add_parser('deepliif')
    deepliif_parser.add_argument("--model_dir", type=str, help="Path to the unserilized model directory", required=True)
    deepliif_parser.add_argument("--tile_size", type=int, help="Size of the tiles to be processed", default=256)
    deepliif_parser.add_argument("--post_processing", type=bool, help="Run postprocessing alogrithem on the results", default=True)
    # Add more deepliif arguments as needed

    return parser.parse_args()


 # type: ignore