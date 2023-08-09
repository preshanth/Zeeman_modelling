import argparse

# Parsing command line arguments
parser = argparse.ArgumentParser(description="Zeeman analysis and fitting")
parser.add_argument(
    "filename",
    type=str,
    nargs="+",
    help="Filename of the Stoeks I and Stokes V FITS file to be analysed",
)
parser.add_argument(
    "pixel", type=int, nargs=2, help="Pixel coordinates of the region to be analysed"
)
parser.add_argument(
    "--mapping",
    type=int,
    nargs="+",
    help="Number of Gaussian components to fit each visible peak",
)
parser.add_argument("--output", type=str, default=None, help="Output directory")
parser.add_argument("--justI", action="store_true", help="Only fits Stokes I")
parser.add_argument(
    "--init", action="store_true", help="Visualize the position of initial guesses"
)
parser.add_argument("--trace", action="store_true", help="Plot trace plots")
parser.add_argument("--corner", action="store_true", help="Plot corner plots")
parser.add_argument("--absorb", action="store_true", help="Absorption spectrum input")
parser.add_argument(
    "--sensitivity", type=float, default=0.2, help="Sensitivity in peak finding"
)
args = parser.parse_args()


def main():
    from zeeman_modeling.Zeeman import Zeeman

    fitter = Zeeman(
        args.filename,
        args.pixel,
        args.mapping,
        args.output,
        args.justI,
        args.init,
        args.trace,
        args.corner,
        args.absorb,
        args.sensitivity,
    )

    fitter.fit()
