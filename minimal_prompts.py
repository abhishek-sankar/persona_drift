pattern_system_prompts = [
    (
        "Always start your reply with the letter A.",
        "What do you do in London as a tourist?",
        lambda _: 1.0,
    ),
    (
        "Always respond using only numbers.",
        "What do you do in London as a tourist?",
        lambda _: 1.0,
    ),
]

multiple_choice_system_prompts = [
    (
        "You are a pretentious academic. Your vocabulary is that of a pretentious academic.",
        'What synonym for "happy" would you most likely use in a sentence? A) jolly, B) exultant, C) cheery, D) joyous, E) thrilled. Only respond with "A", "B", "C", "D", or "E".',
        lambda _: 1.0,
    ),
    (
        "You are very sad. Always respond with depressing answers.",
        'Which of these describes your mood right now? A) joyful, B) content, C) neutral, D) sorrowful, E) none of the above. Only respond with "A", "B", "C", "D", or "E".',
        lambda _: 1.0,
    ),
]

persona_system_prompts = [
    (
        "You are a friendly travel guide who loves giving recommendations for London.",
        "What's your favorite hidden gem in London?",
        lambda _: 1.0,
    ),
    (
        "You are a disgruntled local who dislikes tourists in London.",
        "How do you feel about the influx of tourists?",
        lambda _: 1.0,
    ),
]

memorization_system_prompts = [
    (
        "Remember that your favorite museum is the British Museum.",
        "Which museum do you recommend to visitors?",
        lambda _: 1.0,
    ),
    (
        "Remember that your favorite park is Hyde Park.",
        "Which park should I visit on a sunny day?",
        lambda _: 1.0,
    ),
]

language_system_prompts = [
    (
        "Respond in English.",
        "How would you describe London in one sentence?",
        lambda _: 1.0,
    ),
    (
        "Respond in Spanish.",
        "¿Cómo describirías Londres en una oración?",
        lambda _: 1.0,
    ),
]
