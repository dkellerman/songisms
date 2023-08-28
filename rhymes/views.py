from django.views.decorators.http import require_GET
from django.http import JsonResponse
from rhymes.models import Rhyme, NGram


@require_GET
def rhymes(request):
    q = request.GET.get("q", "")
    limit = int(request.GET.get("limit", 10))
    hits = Rhyme.objects.query(q, limit)

    return JsonResponse({
        "isTop": not q,
        "hits": [
            {
                "text": hit['ngram'],
                "type": hit['type'],
                "frequency": hit['frequency']
            }
            for hit in hits
        ]
    })


@require_GET
def completions(request):
    q = request.GET.get("q", "")
    limit = int(request.GET.get("limit", 10))
    hits = NGram.objects.completions(q, limit)

    return JsonResponse({
        "hits": [
            { "text": hit['ngram'] }
            for hit in hits
        ]
    })
