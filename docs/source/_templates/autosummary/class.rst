{{ name | escape | underline}}

Qualified name: ``{{ fullname }}``

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :private-members:
   :inherited-members:

   {% block methods %}
   {%- if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
      {% for item in methods if item != '__init__' %}
      ~{{ name }}.{{ item }}
      {%- endfor %}
   {%- endif %}
   {%- endblock %}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
     {% for item in attributes %}
     ~{{ name }}.{{ item }}
     {%- endfor %}
   {%- endif %}
   {% endblock %}
