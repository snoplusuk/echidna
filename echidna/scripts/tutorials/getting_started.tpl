{% extends 'python.tpl'%}

{%- block header -%}
""" Python version of getting started tutorial.

This file has been generated automatically by running::

    (ENV) $ jupyter nbconvert --to python --template getting_started
getting_started.ipynb
"""
import matplotlib.pyplot as plt
{% endblock header -%}

## Comment out magic cells and...
## ...Indent all code input cells by 4 spaces
## --> we want them to be inside if __name__ == "__main__"
{% block input %}
{# - sign removes trailing whitespace#}
{%- if cell['metadata'].get('magic', {}) -%}
{%- for line in cell.source.split("\n") -%}
# {{ line }}
{% endfor -%}
{%- else -%}
{%- for line in cell.source.split("\n") -%}
   {{ line }}
{% endfor -%}
{% endif -%}
{% endblock input %}

## Remove one blank line from end
{% block footer -%}
{%- endblock footer %}
