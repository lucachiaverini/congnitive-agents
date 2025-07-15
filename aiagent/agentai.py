import numpy as np
import yaml
import random

WORDS_PER_PAGE = 300

class RAGHumanAgent:
    def __init__(self, profile, config, is_late=False, current_stress=0.0, avg_complexity=None):
        self.profile = profile
        self.config = config
        self.is_late = is_late
        self.current_stress = current_stress
        self.avg_complexity = avg_complexity

        gen = config['general_parameters']
        ai = config['ai_agent']
        self.avg_clicks_to_find_doc = gen['avg_clicks_to_find_doc']
        self.avg_nav_time_per_click_sec = gen.get('avg_nav_time_per_click_sec', 1)
        self.avg_doc_open_time_sec = gen.get('avg_doc_open_time_sec', 2)
        self.avg_read_speed_words_per_sec = gen['avg_read_speed_words_per_sec']
        self.avg_cognitive_processing_sec = gen['avg_cognitive_processing_sec']
        self.avg_writing_speed_words_per_sec = gen['avg_writing_speed_words_per_sec']
        self.doc_type_count = gen['doc_type_count']
        self.min_docs = gen['min_documents_per_operation']
        self.max_docs = gen['max_documents_per_operation']

        # Parametri AI migliorativi
        self.context_knowledge_factor = ai['context_knowledge_factor']
        self.error_rate = ai['error_rate']
        self.hallucination_rate = ai['hallucination_rate']
        self.cognitive_load_tolerance = ai['cognitive_load_tolerance']
        self.short_term_memory_capacity = ai['short_term_memory_capacity']
        self.confidence = ai['confidence']
        self.stress_tolerance = ai['stress_tolerance']
        self.retrieval_latency_sec = ai['retrieval_latency_sec']
        self.generation_latency_sec = ai['generation_latency_sec']

        self.hourly_rate = 35  # ipotetico costo orario umano+AI

        self.document_sources = list(config['document_sources'].keys())
        self.doc_source_probs = [
            config['document_sources'][k]['probability'] for k in self.document_sources
        ]
        self.doc_sources_config = config['document_sources']

        # Wait times opzionale
        self.wait_times = config.get('wait_times_sec', {})

        if self.is_late:
            self.avg_nav_time_per_click_sec *= 1.05
            self.avg_read_speed_words_per_sec /= 1.05
            self.avg_cognitive_processing_sec *= 1.05
            self.error_rate *= 1.05
            self.avg_clicks_to_find_doc *= 1.05

    def simulate_search(self, response_words):
        total_wait_time_sec = 0
        total_navigation_sec = 0
        total_open_doc_sec = 0
        total_read_sec = 0
        total_processing_sec = 0
        errors_total = 0
        hallucinations_total = 0
        documents_details = []

        num_documents = np.random.randint(self.min_docs, self.max_docs + 1)

        for i in range(num_documents):
            # scegli la sorgente
            source = np.random.choice(self.document_sources, p=self.doc_source_probs)
            source_conf = self.doc_sources_config[source]

            # estrai complessità e dimensione, riduci la complessità grazie al RAG
            original_complexity = self.avg_complexity
            complexity = max(1, original_complexity * 0.7)  # RAG riduce la complessità percepita

            num_pages = np.random.randint(
                source_conf['pages_range'][0],
                source_conf['pages_range'][1] + 1
            )
            doc_words = num_pages * WORDS_PER_PAGE

            wait_time_sec = 0
            if source in ["teams", "email"] and self.wait_times:
                w = self.wait_times.get(source, {'min': 0, 'max': 0})
                wait_time_sec = np.random.uniform(w['min'], w['max'])

            effective_error_rate = min(
                self.error_rate + max(0, self.current_stress - self.stress_tolerance), 1.0
            )
            repeats = 1
            if np.random.rand() < effective_error_rate:
                repeats += 1
            if self.confidence < 0.6:
                repeats += 1

            clicks = np.random.normal(self.avg_clicks_to_find_doc, 0.5)
            time_navigation = clicks * self.avg_nav_time_per_click_sec * repeats
            time_open_doc = self.avg_doc_open_time_sec * repeats

            # calcola moltiplicatore complessità
            load_multiplier = 1.0 + (complexity / 10.0)

            memory_overload = False
            doc_complexity_factor = complexity * num_pages
            if doc_complexity_factor > self.short_term_memory_capacity:
                memory_overload = True
                load_multiplier *= 1.2  # meno penalità grazie al RAG

            effective_read_speed = self.avg_read_speed_words_per_sec * (1 + self.context_knowledge_factor)
            time_read_doc = (doc_words / effective_read_speed) * repeats * load_multiplier
            time_processing = self.avg_cognitive_processing_sec * repeats * load_multiplier

            # Latenza AI
            retrieval_time = self.retrieval_latency_sec
            generation_time = self.generation_latency_sec

            total_wait_time_sec += wait_time_sec + retrieval_time + generation_time
            total_navigation_sec += time_navigation
            total_open_doc_sec += time_open_doc
            total_read_sec += time_read_doc
            total_processing_sec += time_processing
            errors_total += 1 if repeats > 1 else 0
            hallucinations_total += 1 if np.random.rand() < self.hallucination_rate else 0

            documents_details.append({
                "doc_number": i + 1,
                "document_source": source,
                "complexity": round(complexity, 2),
                "original_complexity": round(original_complexity, 2),
                "num_pages": int(num_pages),
                "doc_words": int(doc_words),
                "wait_time_min": round(wait_time_sec / 60, 2),
                "navigation_time_min": round(time_navigation / 60, 2),
                "open_doc_time_min": round(time_open_doc / 60, 2),
                "read_doc_time_min": round(time_read_doc / 60, 2),
                "processing_time_min": round(time_processing / 60, 2),
                "retrieval_time_sec": retrieval_time,
                "generation_time_sec": generation_time,
                "errors": 1 if repeats > 1 else 0,
                "hallucination": 1 if hallucinations_total else 0,
                "memory_overload": memory_overload
            })

        time_write_response_sec = response_words / self.avg_writing_speed_words_per_sec

        total_time_sec = (
            total_wait_time_sec
            + total_navigation_sec
            + total_open_doc_sec
            + total_read_sec
            + total_processing_sec
            + time_write_response_sec
        )

        total_time_min = total_time_sec / 60
        cost_eur = (total_time_sec / 3600) * self.hourly_rate

        result = {
            "profile": self.profile,
            "num_documents_consulted": num_documents,
            "total_time_min": round(total_time_min, 2),
            "total_cost_eur": round(cost_eur, 2),
            "total_errors": errors_total,
            "total_hallucinations": hallucinations_total,
            "avg_doc_complexity": round(
                np.mean([doc["complexity"] for doc in documents_details]), 2
            ) if documents_details else 0,
            "time_write_response_min": round(time_write_response_sec / 60, 2),
            "documents_details": documents_details
        }

        return result

# Esempio d'uso:
if __name__ == "__main__":
    with open("config-ai.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    agent = RAGHumanAgent("rag_user", config)
    result = agent.simulate_search(response_words=600)
    print(result)