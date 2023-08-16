#!/usr/bin/env python ./manage.py script

from api.models import Rhyme
from api.utils.ml import gpt_json_query
from songisms import settings

settings.USE_QUERY_CACHE = False

def get_temperature(queries, limit):
    groups = []
    for q in queries:
        result = Rhyme.objects.query(q=q, limit=limit)
        if len(result) > 0:
            rhymes = [q] + [ r['ngram'] for r in result ]
            groups.append(rhymes)

    lines = [ ';'.join(x) for x in groups ]
    formatted_list = '\n'.join(['- ' + l for l in lines])

    scores = gpt_json_query(
        system_message="""You are are helpful songwriting teacher.""",
        user_message="""
            I'm going to give you a few sets of rhymes (each set on one line,
            separated by a semicolon), and for each line I want you to give a number
            between 0 and 1, with 0 indicating that the rhymes are all perfect,
            and 1 indicating that nothing rhymes at all. It's not a strict count of words
            that rhyme, it's more of a general sense of how creative vs. strict the rhymes are -
            what might be called "temperature" when applied to an AI such as yourself.

            IMPORTANT: Output a JSON array with the numbers for each line and nothing else.

            Additonal notes:
                * Be harsh on lines with very few reasonable rhymes, score them closer to zero
            ```
            %s
            ```
        """ % formatted_list,
        model="gpt-4"
    )

    qtemps = list(zip([ w[0] for w in groups ], scores))
    avg_temp = sum([ float(t) for _, t in qtemps ]) / len(qtemps)
    return qtemps, avg_temp


if __name__ == "__main__":
    queries = [
        "in it", "saturday", "let me know", "love story", "one and the same", "use me",
        "a lot", "lovingly", "99 problems", "one more", "shadow", "empty", "corner",
        "feel good", "roll it up", "semblance"
    ]

    qtemps, avg_temp = get_temperature(queries, 50)
    output = [ '__AVG__, %s' % str(avg_temp)] + \
             [ ', '.join([ str(v) for v in x ]) for x in qtemps ]

    print('\n'.join(output))
