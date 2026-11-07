"""
TO be use for:
1. IS_HISTORY_REQUIRED
2. IS_REDIS_HITS_REQUIRED
3. IS_RESPONSE_ANSWERS_QUERY
4. SHOULD_STOP
"""

import ollama

response = ollama.systemone(
    model='laya',
    state='Hello World',
    questions={
        'says_hello': {
            'type': 'noul',
            'instructions': 'Does the state text contain a greeting?',
            'criteria': {
                'true': 'The state text contains a greeting.',
                'false': 'The state text does not contain a greeting.',
            },
        },
    },
)
print(response.answers['says_hello'].noul)