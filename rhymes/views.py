import json
from django import http
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from rhymes.models import Rhyme, NGram, Vote
from wonderwords import RandomWord


@require_http_methods(["GET"])
def rhymes(request):
    q = request.GET.get("q", "")
    limit = min(200, int(request.GET.get("limit", 10)))
    voter_uid = request.GET.get("voter_uid", None)

    if q:
        hits = Rhyme.objects.query(q, 0, limit, voter_uid=voter_uid)

        return http.JsonResponse({
            "isTop": False,
            "hits": [
                {
                    "text": hit['ngram'],
                    "type": hit['type'],
                    "frequency": hit['frequency'],
                    "score": hit['score'],
                    "vote": hit.get('vote', None),
                    "source": hit.get('source', None),
                }
                for hit in hits
            ]
        })


    hits = Rhyme.objects.top_rhymes(0, limit)
    return http.JsonResponse({
        "isTop": True,
        "hits": [{ "text": hit['ngram'], "type": "rhyme" } for hit in hits]
    })


@require_http_methods(["GET"])
def completions(request):
    q = request.GET.get("q", "")
    limit = min(50, int(request.GET.get("limit", 10)))
    hits = NGram.objects.completions(q, limit)

    return http.JsonResponse({
        "hits": [{ "text": hit.text } for hit in hits]
    })


@require_http_methods(["GET"])
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


@csrf_exempt
@require_http_methods(["POST"])
def vote(request):
    data = json.loads(request.body)
    anchor = data.get("anchor")  # required
    label = data.get("label")  # required
    voter_uid = data.get("voter_uid")  # required for now
    remove = data.get("remove", False)
    alt1 = data.get("alt1", None)
    alt2 = data.get("alt2", None)

    if remove and (not anchor or not voter_uid):
        return http.HttpResponseBadRequest("anchor, voter_uid required")
    if not remove and (not anchor or not voter_uid or not label):
        return http.HttpResponseBadRequest("anchor, label, voter_uid required")
    if not alt1 and not alt2:
        return http.HttpResponseBadRequest("specify alt1 or alt1/alt2")
    if alt2 and not alt1:
        return http.HttpResponseBadRequest("must specify alt1 with alt2 present")
    if not alt1 and not alt2 and label not in ('good', 'bad'):
        return http.HttpResponseBadRequest("must vote good/bad without alt2 present")
    if alt1 and alt2 and label in ('good', 'bad'):
        return http.HttpResponseBadRequest("invalid label for alt1/alt2 pair")
    if remove and not remove in ("all", "last"):
        return http.HttpResponseBadRequest("specify remove 'all' or 'last'")

    if remove:
        votes = Vote.objects.filter(anchor=anchor, alt1=alt1, alt2=alt2, voter_uid=voter_uid)
        if votes.exists():
            if remove == "all":
                votes.delete()
            else:
                votes.order_by('-created').last().delete()
        return http.HttpResponse(status=204)
    else:
        Vote.objects.create(anchor=anchor, alt1=alt1, alt2=alt2, label=label,
                            voter_uid=voter_uid)
        return http.HttpResponse(status=201)
