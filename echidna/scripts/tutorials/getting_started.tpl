{% extends 'python.tpl'%}

{%- block header -%}
""" Python version of getting started tutorial.

Please see https://github.com/snoplusuk/echidna/wiki/GettingStarted or the
jupyter notebook form of this tutorial for further details.

This script:
 * Creates :class:`echidna.core.spectra.Spectra` instance
 * Fills `Spectra`
 * Plots `Spectra`
 * Applies cuts and smears `Spectra`
 * Other `Spectra` manipulations e.g. `shrink_to_roi`, `rebin` and
   `scale`

This file has been generated automatically by running::

    (ENV) $ jupyter nbconvert --to python --template getting_started
getting_started.ipynb

Examples:
  To run (from the base directory)::

    $ python echidna/scripts/tutorials/getting_started.py

"""
if __name__ == "__main__":  # for running as a standalone python script too!
    import matplotlib.pyplot as plt
{% endblock header -%}

## Comment out magic cells and...
## ...Indent all code input cells by 4 spaces
## --> we want them to be inside if __name__ == "__main__"
{% block input %}
{# - sign removes trailing whitespace#}
{%- if cell['metadata'].get('magic', {}) -%}
{{ cell.source|comment_lines() }}
{% else -%}
{%- for line in cell.source.split("\n") -%}
{%- if line != "" -%}
{{ line|indent() }}
{% else -%}
{{ line }}
{% endif -%}
{% endfor -%}
{% endif -%}
{% endblock input %}

## Wrap markdown cells
{% block markdowncell %}
{{ cell.source|wrap_text(width=76)|comment_lines() }}
{% endblock markdowncell %}

## Remove one blank line from end
{% block footer -%}
{%- endblock footer %}
