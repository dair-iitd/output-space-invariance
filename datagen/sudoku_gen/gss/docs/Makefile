FIGS=NRC.asy NRC_solved.asy STD.asy NRCX.asy CUBE.asy jigsaw.asy jigsawxl.asy bigjigcube.asy 3dcube.asy parquet.asy linked.asy

PDFC=pdflatex
PDFFLAGS=-interaction nonstopmode -halt-on-error
HTMC=htlatex
HTMFLAGS="" "" "" "-interaction=nonstopmode"

pdf: gss.tex pdffigs
	$(PDFC) $(PDFFLAGS) gss.tex
	$(PDFC) $(PDFFLAGS) gss.tex
html: index.tex svgfigs
	$(HTMC) index.tex $(HTMFLAGS)
	$(HTMC) index.tex $(HTMFLAGS)
# some post processing
	sed -i 's/\(<head><title>\)\(<\/title>\)/\1Generic Sudoku Solver\2\n<meta name="viewport" content="width=device-width, initial-scale=1">/g' index.html
index.tex: gss.tex
# we create index.tex from gss.tex, adding an svg extension to all included images
	sed 's/\(includegraph[^{]*[^}]*\)/\1.svg/g' gss.tex > index.tex
pdffigs: $(FIGS)
# we do not use inline asy scripts because this way I can use svg's or pdf's depending on the target
# with asy it would create eps and then convert those to png's. I prefer svg vector graphics
	asy -f pdf $(FIGS)
	touch pdffigs
svgfigs: $(FIGS)
	asy -f svg $(FIGS)
# this one svg is linked to a large number of small png files. I do not know how to 
# convince asymptote to make a cleaner output so here is a quick 'n dirty script to 
# at least embed those linked png's in the svg
	./embed.py bigjigcube.svg
	rm *.png
	touch svgfigs
clean:latexclean
# cleans up all
	-rm *.pdf *.svg index.*
latexclean:
# cleans up the files latex/htlatex create except for those files that are part of the resulting pdf and html page
	-find . -name 'gss.*' ! -name 'gss.tex' ! -name 'gss.pdf' -type f -exec rm -f {} +
	-find . -name 'index.*' ! -name 'index.html' ! -name 'index.css' -type f -exec rm -f {} +
	-find . -name '*.pdf' ! -name 'gss.pdf' -type f -exec rm -f {} +
	-rm pdffigs svgfigs *_.out
	

