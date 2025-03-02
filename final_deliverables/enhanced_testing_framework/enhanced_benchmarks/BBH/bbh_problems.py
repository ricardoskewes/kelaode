"""
Enhanced benchmark problems from the Big-Bench Hard (BBH) dataset.
These problems require multi-step logical reasoning.
"""

BBH_PROBLEMS = [
    {
        "id": "bbh_logical_deduction_1",
        "category": "logical_deduction",
        "problem": "Five friends (Alex, Bob, Charlie, Dave, and Eve) are sitting in a row at the movies. We know that: Alex is not sitting next to Bob or Charlie. Eve is sitting to the right of Charlie and to the left of Alex. Bob is sitting to the left of Dave. Who is sitting in the middle?",
        "benchmark": "BBH",
        "difficulty": "hard"
    },
    {
        "id": "bbh_date_understanding_1",
        "category": "date_understanding",
        "problem": "Today is April 15, 2023. What day of the week will it be 100 days from now?",
        "benchmark": "BBH",
        "difficulty": "medium"
    },
    {
        "id": "bbh_tracking_shuffled_objects_1",
        "category": "tracking_shuffled_objects",
        "problem": "There are 5 cups in a row, labeled 1, 2, 3, 4, and 5 from left to right. The following operations are performed: Swap cups 2 and 4. Swap cups 1 and 5. Swap cups 3 and 1. What is the new ordering of the cups from left to right?",
        "benchmark": "BBH",
        "difficulty": "medium"
    },
    {
        "id": "bbh_dyck_languages_1",
        "category": "dyck_languages",
        "problem": "Check if the following string of brackets is balanced: (()(())())((()))(())",
        "benchmark": "BBH",
        "difficulty": "medium"
    },
    {
        "id": "bbh_word_sorting_1",
        "category": "word_sorting",
        "problem": "Sort the following words alphabetically: 'apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape'",
        "benchmark": "BBH",
        "difficulty": "easy"
    },
    {
        "id": "bbh_navigate_1",
        "category": "navigate",
        "problem": "You start at position (0, 0) facing north. You perform the following sequence of steps: Walk forward 5 steps, turn right, walk forward 3 steps, turn left, walk forward 2 steps, turn around (180 degrees), walk forward 7 steps, turn right, walk forward 1 step. What is your final position and which direction are you facing?",
        "benchmark": "BBH",
        "difficulty": "hard"
    },
    {
        "id": "bbh_reasoning_about_colored_objects_1",
        "category": "reasoning_about_colored_objects",
        "problem": "I have a red ball, a blue ball, a red cube, a blue cube, a red cylinder, and a blue cylinder. I will remove all the cubes and all the blue objects. How many objects will I have left?",
        "benchmark": "BBH",
        "difficulty": "medium"
    }
]
