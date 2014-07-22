(TeX-add-style-hook "densityratio_writeup"
 (lambda ()
    (TeX-add-symbols
     '("abs" 1)
     '("set" 1)
     '("bkt" 1)
     '("prn" 1)
     "RR"
     "yy"
     "YY"
     "rf"
     "te")
    (TeX-run-style-hooks
     "mathrsfs"
     "mathtools"
     "amsthm"
     "amssymb"
     "amsfonts"
     "amsmath"
     "geometry"
     "latex2e"
     "art12"
     "article"
     "12pt")))

