from django.views.decorators.http import require_GET, require_POST
from django import http
from rhymes.models import Rhyme, NGram, Vote
from wonderwords import RandomWord


@require_GET
def rhymes(request):
    q = request.GET.get("q", "")
    limit =  min(200, int(request.GET.get("limit", 10)))

    if q:
        hits = Rhyme.objects.query(q, 0, limit)
        return http.JsonResponse({
            "isTop": False,
            "hits": [
                {
                    "text": hit['ngram'],
                    "type": hit['type'],
                    "frequency": hit['frequency'],
                    "score": hit['score'],
                }
                for hit in hits
            ]
        })


    hits = Rhyme.objects.top_rhymes(0, limit)
    return http.JsonResponse({
        "isTop": True,
        "hits": [{ "text": hit['ngram'], "type": "rhyme" } for hit in hits]
    })


@require_GET
def completions(request):
    q = request.GET.get("q", "")
    limit = min(50, int(request.GET.get("limit", 10)))
    hits = NGram.objects.completions(q, limit)

    return http.JsonResponse({
        "hits": [{ "text": hit.text } for hit in hits]
    })


@require_GET
def rlhf(request):
    limit = min(20, int(request.GET.get("limit", 10)))
    rw = RandomWord()

    return http.JsonResponse({
        "hits": [{
            "anchor": rw.word(),
            "alt1": rw.word(),
            "alt2": rw.word(),
        } for _ in range(limit)]
    })


@require_POST
def vote(request):
    anchor = request.POST.get("anchor")  # required
    label = request.POST.get("label")  # required
    voter_uid = request.POST.get("voter_uid")  # required for now

    if not anchor or not label or not voter_uid:
        return http.HttpResponseBadRequest("anchor, label, voter_uid required")

    alt1 = request.POST.get("alt1", None)
    alt2 = request.POST.get("alt2", None)

    if (alt1 and not alt2) or (alt2 and not alt1):
        return http.HttpResponseBadRequest("specify both alt1 and alt2 or neither")
    if not alt1 and not alt2 and label not in ('good', 'bad'):
        return http.HttpResponseBadRequest("must vote good/bad with no alt1/alt2 pair")
    if alt1 and alt2 and label in ('good', 'bad'):
        return http.HttpResponseBadRequest("invalid label for alt1/alt2 pair")

    Vote.objects.create(anchor=anchor, alt1=alt1, alt2=alt2, label=label,
                        voter_uid=voter_uid)

    return http.HttpResponseCreated()
