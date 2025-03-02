"""
Enhanced benchmark problems from the HotpotQA dataset.
These problems require multi-hop reading comprehension.
"""

HOTPOTQA_PROBLEMS = [
    {
        "id": "hotpotqa_1",
        "category": "multi_hop_qa",
        "problem": "The film 'Inception' was directed by a British-American filmmaker who also directed a trilogy of superhero films featuring which comic book character?",
        "benchmark": "HotpotQA",
        "difficulty": "medium"
    },
    {
        "id": "hotpotqa_2",
        "category": "multi_hop_qa",
        "problem": "The author of 'The Da Vinci Code' also wrote a novel featuring the character Robert Langdon that was adapted into a 2009 film directed by Ron Howard. What is the name of this novel?",
        "benchmark": "HotpotQA",
        "difficulty": "medium"
    },
    {
        "id": "hotpotqa_3",
        "category": "multi_hop_qa",
        "problem": "The scientist who developed the theory of general relativity was born in which country that was later unified with other states in 1871?",
        "benchmark": "HotpotQA",
        "difficulty": "hard"
    },
    {
        "id": "hotpotqa_4",
        "category": "multi_hop_qa",
        "problem": "The 2019 film 'Joker' starred an actor who previously played which Roman emperor in the 2000 film 'Gladiator'?",
        "benchmark": "HotpotQA",
        "difficulty": "hard"
    },
    {
        "id": "hotpotqa_5",
        "category": "multi_hop_qa",
        "problem": "The CEO of Tesla Motors also founded a company that merged with Confinity in 2000. What was the name of this company?",
        "benchmark": "HotpotQA",
        "difficulty": "medium"
    },
    {
        "id": "hotpotqa_6",
        "category": "multi_hop_qa",
        "problem": "The author who wrote 'Pride and Prejudice' had a sister who was also a novelist. What was the name of this sister?",
        "benchmark": "HotpotQA",
        "difficulty": "hard"
    },
    {
        "id": "hotpotqa_7",
        "category": "multi_hop_qa",
        "problem": "The chemical element with atomic number 79 was named after which country whose Latin name is 'Aurum'?",
        "benchmark": "HotpotQA",
        "difficulty": "hard"
    }
]
