{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members: __init__
   :exclude-members: {{ methods|join(', ') }}
   :show-inheritance:

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: .
   {% for item in methods %}
      {%- if item != '__init__' and item not in ['get_params', 'set_params', 'set_output', 'fit_transform', 'get_metadata_routing', 'set_fit_request', 'set_transform_request'] %}
      ~{{ fullname }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
