import time
import json
import random
from django import http
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from rhymes.models import Rhyme, NGram, Vote
from wonderwords import RandomWord


@require_http_methods(["GET"])
def rhymes(request):
    ts = time.time()
    q = request.GET.get("q", "")
    limit = min(200, int(request.GET.get("limit", 10)))
    voter_uid = request.GET.get("voterUid", None)

    if q:
        hits = Rhyme.objects.query(q, 0, limit, voter_uid=voter_uid)
        resp = http.JsonResponse({
            "isTop": False,
            "hits": [
                {
                    "text": hit['ngram'],
                    "type": 'rhyme-l2' if hit['frequency'] == 0 else 'rhyme',
                    "frequency": hit['frequency'],
                    "score": hit['score'],
                    "vote": hit.get('vote', None),
                    "source": hit.get('source', None),
                }
                for hit in hits
            ]
        })
        print(f'rhymes query {q} took {(time.time() - ts) * 1000} ms')
        return resp


    hits = Rhyme.objects.top_rhymes(0, limit)
    resp = http.JsonResponse({
        "isTop": True,
        "hits": [{
            "text": hit['ngram'],
            "type": "rhyme",
            "frequency": hit['rfrequency']
        } for hit in hits]
    })
    print(f'rhymes top took {(time.time() - ts) * 1000} ms')
    return resp


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
    rhymes = Rhyme.objects.exclude(level=1).order_by('?')

    all = []
    for r in rhymes[:limit * 2]:
        all.append(dict(rfrom=r.from_ngram.text, rto=r.to_ngram.text, rto2=r.from_ngram.text))

    hits = []
    for i in range(0, len(all), 2):
        r, r2 = all[i], all[i + 1]
        hit = dict(anchor=r['rfrom'], alt1=r['rto'], alt2=r2['rto'])
        if hit['alt1'] == hit['anchor']:
            hit['alt1'] = r2['rto2']
        elif hit['alt2'] == hit['anchor'] or (hit['alt1'] == hit['alt2']):
            hit['alt2'] = r2['rto2']
        if random.random() > 0.5:
            hit['alt1'], hit['alt2'] = hit['alt2'], hit['alt1']
        hits.append(hit)

    return http.JsonResponse({
        "hits": hits
    })


@csrf_exempt
@require_http_methods(["POST"])
def vote(request):
    data = json.loads(request.body)
    anchor = data.get("anchor")  # required
    label = data.get("label")  # required
    voter_uid = data.get("voterUid")  # required for now
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
