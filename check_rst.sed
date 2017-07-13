## Sed-file to clean-up README.rst from Sphinx-only constructs,
##   so that *PyPi* can format it properly.
##   To check for remaining errors, install ``sphinx`` and run:
##
##          sed -f "this_file.txt" README.rst | rst2html.py  --halt=warning
##

## Selected Sphinx-only Roles.
#
s/:abbr:`\([^`]*\)`/\1/gi
s/:ref:`\([^`]*\)`/`\1`_/gi
s/:term:`\([^`]*\)`/**\1**/gi
s/:dfn:`\([^`]*\)`/**\1**/gi
s/:\(samp\|guilabel\|menuselection\):`\([^`]*\)`/``\1``/gi


## Sphinx-only roles:
#        :foo:`bar` --> foo(``bar``)
#
s/:\([a-z]*\):`\([^`]*\)`/\1(``\2``)/gi


## Sphinx-only Directives.
#
s/\.\. +doctest/code-block/i
s/\.\. +plot/raw/i
s/\.\. +seealso/info/i
s/\.\. +glossary/rubric/i
s/\.\. +figure::/../i


## Other
#
s/|version|/x.x.x/gi
