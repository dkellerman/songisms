# patch for django-graphene 2.x :(
import django
from django.utils.encoding import force_str
django.utils.encoding.force_text = force_str
