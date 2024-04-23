import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Extract areas from WSI slides based on the provided HistoQC masks")
    parser.add_argument("--slides_path", type=str, help="Path to the WSI slides location")
    parser.add_argument("--masks_path", type=str, help="Path to the HistoQC masks location")
    parser.add_argument("--down_sample_rate", type=int, help="the rate of down sampling the extracted regions", default=5)
    parser.add_argument("--output_path", type=str, help="Path to the output location")
    return parser.parse_args()


