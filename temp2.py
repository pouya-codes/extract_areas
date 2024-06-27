import pyvips
import timeit
def run():
    slide = pyvips.Image.thumbnail("CMU-1.ndpi", 500)
    thumb.write_to_file("tn_CMU-1.png")

timeit.timeit(lambda :run(), number=10)