import yaml
import json
import random
import copy
import numpy as np
from cognitiveagent.agent import HumanSearchAgent
from aiagent.agentai import RAGHumanAgent 
from datetime import datetime

def vary_config(base_config):
    """
    Restituisce una versione leggermente modificata di base_config
    per simulare piccoli cambiamenti nell'ambiente.
    Tutte le variazioni sono tra -5% e +5%.
    """
    new_config = copy.deepcopy(base_config)

    # es. variazione ±1-5% su parametri generali
    gen = new_config['general_parameters']
    for key in [
        'avg_clicks_to_find_doc',
        'avg_nav_time_per_click_sec',
        'avg_doc_open_time_sec',
        'avg_read_speed_words_per_sec',
        'avg_cognitive_processing_sec',
        'avg_writing_speed_words_per_sec'
    ]:
        factor = random.uniform(0.95, 1.05)
        gen[key] *= factor

    # varia leggermente anche fatigue
    if 'fatigue' in new_config:
        for k in new_config['fatigue']:
            factor = random.uniform(0.95, 1.05)
            new_config['fatigue'][k] *= factor

    # varia leggermente probabilità delle sorgenti
    total_prob = 0.0
    for src in new_config['document_sources']:
        variation = random.uniform(0.95, 1.05)
        new_config['document_sources'][src]['probability'] *= variation
        total_prob += new_config['document_sources'][src]['probability']

    # normalizza di nuovo le probabilità
    for src in new_config['document_sources']:
        new_config['document_sources'][src]['probability'] /= total_prob

    return new_config

def sample_complexity(complexity_dist, n_docs, complexity_ranges):
    levels = list(complexity_dist.keys())
    probs = np.array(list(complexity_dist.values()), dtype=float)
    probs = probs / probs.sum()
    sampled_levels = np.random.choice(levels, size=n_docs, p=probs)
    complexities = [
        round(np.random.uniform(*complexity_ranges[level]), 2)
        for level in sampled_levels
    ]
    return complexities

def choose_profile(avg_complexity, profile_assignment):
    avg_complexity = round(float(avg_complexity), 2)
    junior_max = float(profile_assignment['junior_max'])
    mid_max = float(profile_assignment['mid_max'])
    # junior: <= junior_max (es. 3.0)
    if avg_complexity <= junior_max:
        return 'junior'
    # mid: > junior_max e <= mid_max (es. 3.01 fino a 7.0)
    elif junior_max < avg_complexity <= mid_max:
        return 'mid'
    # senior: > mid_max
    else:
        return 'senior'

def simulate_tickets(n_tickets, base_config_path, ai_config_path, output_file_human, output_file_ai):
    # carica configurazione base
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    with open(ai_config_path, "r", encoding="utf-8") as f:
        ai_config = yaml.safe_load(f)

    complexity_ranges = {k: tuple(v) for k, v in base_config['complexity_ranges'].items()}
    profile_assignment = base_config['profile_assignment']

    all_results_human = []
    all_results_ai = []

    for i in range(n_tickets):
        # varia leggermente la config
        config = vary_config(base_config)
        ai_config_var = vary_config(ai_config)  # puoi variare anche la config AI se vuoi

        # --- Estrazione profilo pesata ---
        if 'organizations' in config and 'profiles' in config['organizations']:
            pop_dist = config['organizations']['profiles']
            weighted_profiles = []
            for prof, n in pop_dist.items():
                weighted_profiles.extend([prof] * n)
            profile = random.choice(weighted_profiles)
        else:
            profiles_list = list(config['profiles'].keys())
            profile = random.choice(profiles_list)
        # ----------------------------------

        is_late = random.random() < 0.3
        current_stress = round(random.uniform(0.0, 1.0), 2)
        response_words = random.randint(500, 1000)

        # 1. Genera i documenti e la loro complessità
        n_docs = random.randint(config['general_parameters']['min_documents_per_operation'],
                                config['general_parameters']['max_documents_per_operation'])
        complexity_dist = config['task']['complexity_distribution']
        complexities = sample_complexity(complexity_dist, n_docs, complexity_ranges)
        avg_complexity = np.mean(complexities)
        profile = choose_profile(avg_complexity, profile_assignment)

        # --- Simulazione Human ---
        agent_human = HumanSearchAgent(profile, config, is_late, current_stress, avg_complexity)
        result_human = agent_human.simulate_search(response_words)
        result_human["is_late"] = is_late
        result_human["current_stress"] = current_stress
        result_human["response_words"] = response_words
        all_results_human.append(result_human)

        # --- Simulazione AI (RAG) ---
        agent_ai = RAGHumanAgent(profile, ai_config, is_late, current_stress, avg_complexity)
        result_ai = agent_ai.simulate_search(response_words)
        result_ai["is_late"] = is_late
        result_ai["current_stress"] = current_stress
        result_ai["response_words"] = response_words
        all_results_ai.append(result_ai)

        print(f"Simulazione ticket {i+1} completata")

    # salva tutti i JSON in due file distinti
    with open(output_file_human, "w", encoding="utf-8") as f:
        json.dump(all_results_human, f, indent=2, ensure_ascii=False)
    with open(output_file_ai, "w", encoding="utf-8") as f:
        json.dump(all_results_ai, f, indent=2, ensure_ascii=False)

    print(f"\nSalvato output di {n_tickets} simulazioni in {output_file_human} e {output_file_ai}")

if __name__ == "__main__":
    n_tickets = 10000
    base_config_path = "config.yaml"
    ai_config_path = "config-ai.yaml"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file_human = f"simulation_results-human.json"
    output_file_ai = f"simulation_results-ai.json"

    simulate_tickets(n_tickets, base_config_path, ai_config_path, output_file_human, output_file_ai)