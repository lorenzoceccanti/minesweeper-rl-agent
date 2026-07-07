$pdf_mode = 1;
$out_dir = 'build';
$aux_dir = 'build';
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build %O %S';

# biblatex/biber support: latexmk auto-detects biber runs from .bcf files,
# but we pin the biber output directory explicitly to keep everything in build/.
$biber = 'biber --output-directory build %O %S';

# Keep auxiliary/build clutter fully out of report/, only report/build/ is touched.
$clean_ext = 'bbl bcf run.xml nav snm vrb synctex.gz';
