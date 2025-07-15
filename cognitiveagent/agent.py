import numpy as np
import yaml
import json
import random


WORDS_PER_PAGE = 300

class HumanSearchAgent:
    def __init__(self, profile, config, is_late=False, current_stress=0.0, avg_complexity=None):
        self.profile = profile
        self.config = config
        self.is_late = is_late
        self.current_stress = current_stress
        self.avg_complexity = avg_complexity

        gen = config['general_parameters']
        self.avg_clicks_to_find_doc = gen['avg_clicks_to_find_doc']
        self.avg_nav_time_per_click_sec = gen['avg_nav_time_per_click_sec']
        self.avg_doc_open_time_sec = gen['avg_doc_open_time_sec']
        self.avg_read_speed_words_per_sec = gen['avg_read_speed_words_per_sec']
        self.avg_cognitive_processing_sec = gen['avg_cognitive_processing_sec']
        self.avg_writing_speed_words_per_sec = gen['avg_writing_speed_words_per_sec']
        self.doc_type_count = gen['doc_type_count']
        self.min_docs = gen['min_documents_per_operation']
        self.max_docs = gen['max_documents_per_operation']

        prof = config['profiles'][profile]
        self.hourly_rate = prof['hourly_rate']
        self.context_knowledge_factor = prof['context_knowledge_factor']
        self.error_rate = prof['error_rate']
        self.cognitive_load_tolerance = prof['cognitive_load_tolerance']
        self.short_term_memory_capacity = prof['short_term_memory_capacity']
        self.confidence = prof['confidence']
        self.stress_tolerance = prof['stress_tolerance']

        fatigue = gen.get('fatigue', {})
        self.fatigue_nav = fatigue.get('nav_speed_factor', 1.0)
        self.fatigue_read = fatigue.get('read_speed_factor', 1.0)
        self.fatigue_processing = fatigue.get('processing_delay_factor', 1.0)
        self.fatigue_error_rate = fatigue.get('error_rate_factor', 1.0)
        self.fatigue_retries = fatigue.get('retries_factor', 1.0)

        self.document_sources = list(config['document_sources'].keys())
        self.doc_source_probs = [
            config['document_sources'][k]['probability'] for k in self.document_sources
        ]
        self.doc_sources_config = config['document_sources']

        self.wait_times = config['wait_times_sec']

        if self.is_late:
            self.avg_nav_time_per_click_sec *= self.fatigue_nav
            self.avg_read_speed_words_per_sec /= self.fatigue_read
            self.avg_cognitive_processing_sec *= self.fatigue_processing
            self.error_rate *= self.fatigue_error_rate
            self.avg_clicks_to_find_doc *= self.fatigue_retries

    def simulate_search(self, response_words):
        total_wait_time_sec = 0
        total_navigation_sec = 0
        total_open_doc_sec = 0
        total_read_sec = 0
        total_processing_sec = 0
        errors_total = 0
        documents_details = []

        num_documents = np.random.randint(self.min_docs, self.max_docs + 1)

        for i in range(num_documents):
            # scegli la sorgente
            source = np.random.choice(self.document_sources, p=self.doc_source_probs)
            source_conf = self.doc_sources_config[source]

            # estrai complessità e dimensione
            complexity = self.avg_complexity
            
            num_pages = np.random.randint(
                source_conf['pages_range'][0],
                source_conf['pages_range'][1] + 1
            )
            doc_words = num_pages * WORDS_PER_PAGE

            wait_time_sec = 0
            if source in ["teams", "email"]:
                w = self.wait_times[source]
                wait_time_sec = np.random.uniform(w['min'], w['max'])

            effective_error_rate = min(
                self.error_rate + max(0, self.current_stress - self.stress_tolerance), 1.0
            )
            repeats = 1
            if np.random.rand() < effective_error_rate:
                repeats += 1
            if self.confidence < 0.6:
                repeats += 1

            clicks = np.random.normal(self.avg_clicks_to_find_doc, 1)
            time_navigation = clicks * self.avg_nav_time_per_click_sec * repeats
            time_open_doc = self.avg_doc_open_time_sec * repeats

            # calcola moltiplicatore complessità
            load_multiplier = 1.0 + (complexity / 10.0)

            memory_overload = False
            doc_complexity_factor = complexity * num_pages
            if doc_complexity_factor > self.short_term_memory_capacity:
                memory_overload = True
                load_multiplier *= 1.5

            effective_read_speed = self.avg_read_speed_words_per_sec * (1 + self.context_knowledge_factor)
            time_read_doc = (doc_words / effective_read_speed) * repeats * load_multiplier
            time_processing = self.avg_cognitive_processing_sec * repeats * load_multiplier

            total_wait_time_sec += wait_time_sec
            total_navigation_sec += time_navigation
            total_open_doc_sec += time_open_doc
            total_read_sec += time_read_doc
            total_processing_sec += time_processing
            errors_total += 1 if repeats > 1 else 0

            documents_details.append({
                "doc_number": i + 1,
                "document_source": source,
                "complexity": round(complexity, 2),
                "num_pages": int(num_pages),
                "doc_words": int(doc_words),
                "wait_time_min": round(wait_time_sec / 60, 2),
                "navigation_time_min": round(time_navigation / 60, 2),
                "open_doc_time_min": round(time_open_doc / 60, 2),
                "read_doc_time_min": round(time_read_doc / 60, 2),
                "processing_time_min": round(time_processing / 60, 2),
                "errors": 1 if repeats > 1 else 0,
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
            "avg_doc_complexity": round(
                np.mean([doc["complexity"] for doc in documents_details]), 2
            ) if documents_details else 0,
            "time_write_response_min": round(time_write_response_sec / 60, 2),
            "documents_details": documents_details
        }

        return result