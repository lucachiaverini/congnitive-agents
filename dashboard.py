import gradio as gr
import pandas as pd
import plotly.express as px
import json
import yaml

# --- Caricamento dati una sola volta ---
DATA_PATH_HUMAN = "./simulation_results-human.json"
DATA_PATH_AI = "./simulation_results-ai.json"

with open(DATA_PATH_HUMAN, "r", encoding="utf-8") as f:
    DATA_HUMAN = json.load(f)
with open(DATA_PATH_AI, "r", encoding="utf-8") as f:
    DATA_AI = json.load(f)

DF_HUMAN = pd.DataFrame([d for d in DATA_HUMAN if isinstance(d, dict)])
DF_HUMAN["agent_type"] = "Human"
DF_AI = pd.DataFrame([d for d in DATA_AI if isinstance(d, dict)])
DF_AI["agent_type"] = "AI"

DF_ALL = pd.concat([DF_HUMAN, DF_AI], ignore_index=True)

# Calcola i massimi una sola volta
YMAX_TIME = DF_HUMAN["total_time_min"].max()
YMAX_COST = DF_HUMAN["total_cost_eur"].max()

def plot_cost_per_profile():
    df = DF_HUMAN
    tickets = df[df["profile"].isin(["junior", "mid", "senior"]) & df["total_cost_eur"].notnull()]
    if tickets.empty:
        return "Nessun ticket valido trovato nei dati."
    fig = px.box(
        tickets,
        x="profile",
        y="total_cost_eur",
        color="profile",
        points="all",
        title="Distribuzione costo totale per ticket per profilo",
        labels={
            "profile": "Profilo",
            "total_cost_eur": "Costo totale ticket (€)"
        },
        category_orders={"profile": ["junior", "mid", "senior"]},
        range_y=[0, YMAX_COST],
        template="plotly_dark"
    )
    return fig

def plot_cost_per_profile_ai():
    df = DF_AI
    tickets = df[df["profile"].isin(["junior", "mid", "senior"]) & df["total_cost_eur"].notnull()]
    if tickets.empty:
        return "Nessun ticket valido trovato nei dati."
    fig = px.box(
        tickets,
        x="profile",
        y="total_cost_eur",
        color="profile",
        points="all",
        title="Distribuzione costo totale per ticket per profilo (AI)",
        labels={
            "profile": "Profilo",
            "total_cost_eur": "Costo totale ticket (€)"
        },
        category_orders={"profile": ["junior", "mid", "senior"]},
        range_y=[0, YMAX_COST],
        template="plotly_dark"
    )
    return fig

def boxplot_time_vs_complexity_per_profile():
    df = DF_HUMAN
    tickets = df[df["profile"].isin(["junior", "mid", "senior"]) & df["total_time_min"].notnull()]
    if tickets.empty:
        return "Nessun ticket valido trovato nei dati."
    fig = px.box(
        tickets,
        x="profile",
        y="total_time_min",
        color="profile",
        points="all",
        title="Distribuzione tempo di chiusura per ticket per profilo",
        labels={
            "profile": "Profilo",
            "total_time_min": "Tempo chiusura ticket (min)"
        },
        category_orders={"profile": ["junior", "mid", "senior"]},
        range_y=[0, YMAX_TIME],
        template="plotly_dark"
    )
    return fig

def boxplot_time_vs_complexity_per_profile_ai():
    df = DF_AI
    tickets = df[df["profile"].isin(["junior", "mid", "senior"]) & df["total_time_min"].notnull()]
    if tickets.empty:
        return "Nessun ticket valido trovato nei dati."
    fig = px.box(
        tickets,
        x="profile",
        y="total_time_min",
        color="profile",
        points="all",
        title="Distribuzione tempo di chiusura per ticket per profilo (AI)",
        labels={
            "profile": "Profilo",
            "total_time_min": "Tempo chiusura ticket (min)"
        },
        category_orders={"profile": ["junior", "mid", "senior"]},
        range_y=[0, YMAX_TIME],
        template="plotly_dark"
    )
    return fig

def boxplot_cost_vs_complexity_per_profile():
    df = DF_HUMAN
    tickets = df[df["profile"].isin(["junior", "mid", "senior"]) & df["total_cost_eur"].notnull()]
    if tickets.empty:
        return "Nessun ticket valido trovato nei dati."
    fig = px.box(
        tickets,
        x="profile",
        y="total_cost_eur",
        color="profile",
        points="all",
        title="Distribuzione costo totale per ticket per profilo",
        labels={
            "profile": "Profilo",
            "total_cost_eur": "Costo totale ticket (€)"
        },
        category_orders={"profile": ["junior", "mid", "senior"]},
        range_y=[0, YMAX_COST],
        template="plotly_dark"
    )
    return fig

def boxplot_cost_vs_complexity_per_profile_ai():
    df = DF_AI
    tickets = df[df["profile"].isin(["junior", "mid", "senior"]) & df["total_cost_eur"].notnull()]
    if tickets.empty:
        return "Nessun ticket valido trovato nei dati."
    fig = px.box(
        tickets,
        x="profile",
        y="total_cost_eur",
        color="profile",
        points="all",
        title="Distribuzione costo totale per ticket per profilo (AI)",
        labels={
            "profile": "Profilo",
            "total_cost_eur": "Costo totale ticket (€)"
        },
        category_orders={"profile": ["junior", "mid", "senior"]},
        range_y=[0, YMAX_COST],
        template="plotly_dark"
    )
    return fig

def get_ticket_revenue(avg_complexity, ticket_revenue_cfg):
    for level, info in ticket_revenue_cfg.items():
        min_r, max_r = info['range']
        if min_r <= avg_complexity <= max_r:
            return info['revenue']
    return 0  # fallback

def compute_statistics_dict(df, data, label=""):
    # Carica dati economici dal config YAML
    with open("config-economics-kpi.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    economics = config["economics"]
    ticket_revenue_cfg = economics["ticket_revenue"]
    investment = economics["investment"]
    equity = economics["equity"]
    discount_rate = economics["discount_rate"]
    periods = economics["periods"]  # es: 5
    taxes = economics["taxes"]

    # Calcola il revenue per ogni ticket in base alla sua complessità media
    revenues = []
    for _, row in df.iterrows():
        avg_complexity = row.get("avg_doc_complexity", 0)
        revenue = get_ticket_revenue(avg_complexity, ticket_revenue_cfg)
        revenues.append(revenue)
    df["revenue"] = revenues

    # --- Calcoli annualizzati ---
    total_tickets = len(df)
    total_revenue = df["revenue"].sum()
    total_cost = df["total_cost_eur"].sum()
    operating_income = total_revenue - total_cost
    net_income = operating_income * (1 - taxes)

    # Annualizza i valori
    tickets_per_year = total_tickets / periods
    revenue_per_year = total_revenue / periods
    cost_per_year = total_cost / periods
    operating_income_per_year = operating_income / periods
    net_income_per_year = net_income / periods

    # KPI annualizzati
    roi = (net_income_per_year / investment) * 100 if investment else 0
    ros = (operating_income_per_year / revenue_per_year) * 100 if revenue_per_year else 0
    roe = (net_income_per_year / equity) * 100 if equity else 0

    # NPV su 5 anni (già corretto)
    cash_flow = net_income_per_year
    npv = -investment
    for t in range(1, periods + 1):
        npv += cash_flow / ((1 + discount_rate) ** t)

    # Statistiche base ticket
    total_tickets = len(df)
    tickets_per_profile = df["profile"].value_counts().to_dict()
    avg_time = df["total_time_min"].mean()
    min_time = df["total_time_min"].min()
    max_time = df["total_time_min"].max()
    avg_cost = df["total_cost_eur"].mean()
    min_cost = df["total_cost_eur"].min()
    max_cost = df["total_cost_eur"].max()
    avg_errors = df["total_errors"].mean()
    late_pct = 100 * df["is_late"].sum() / total_tickets if total_tickets else 0
    avg_docs = df["num_documents_consulted"].mean()
    avg_words = df["response_words"].mean()
    avg_stress = df["current_stress"].mean()
    avg_write_time = df["time_write_response_min"].mean()

    # Statistiche documenti
    docs = []
    for t in data:
        docs.extend(t.get("documents_details", []))
    df_docs = pd.DataFrame(docs)
    if not df_docs.empty:
        avg_doc_complexity = df_docs["complexity"].mean()
        avg_doc_pages = df_docs["num_pages"].mean()
        avg_doc_words = df_docs["doc_words"].mean()
        avg_doc_read = df_docs["read_doc_time_min"].mean()
        avg_doc_open = df_docs["open_doc_time_min"].mean()
        avg_doc_nav = df_docs["navigation_time_min"].mean()
        avg_doc_wait = df_docs["wait_time_min"].mean()
        avg_doc_proc = df_docs["processing_time_min"].mean()
        avg_doc_errors = df_docs["errors"].mean()
        pct_doc_memover = 100 * df_docs["memory_overload"].sum() / len(df_docs)
        doc_sources = df_docs["document_source"].value_counts(normalize=True).to_dict()
    else:
        avg_doc_complexity = avg_doc_pages = avg_doc_words = avg_doc_read = avg_doc_open = avg_doc_nav = avg_doc_wait = avg_doc_proc = avg_doc_errors = pct_doc_memover = 0
        doc_sources = {}

    # --- KPI ECONOMICI ---
    total_cost = df["total_cost_eur"].sum()
    operating_income = total_revenue - total_cost
    net_income = operating_income * (1 - taxes)
    roi = (net_income / investment) * 100 if investment else 0
    ros = (operating_income / total_revenue) * 100 if total_revenue else 0
    roe = (net_income / equity) * 100 if equity else 0
    cash_flow = net_income / periods if periods else 0
    npv = -investment
    for t in range(1, periods + 1):
        npv += cash_flow / ((1 + discount_rate) ** t)

    # Ritorna dizionari per ogni sezione
    return {
        "kpi": {
            f"Ricavi totali (5 anni){label}": f"€{total_revenue:,.2f}",
            f"Ricavi medi annui{label}": f"€{revenue_per_year:,.2f}",
            f"Costi totali (5 anni){label}": f"€{total_cost:,.2f}",
            f"Costi medi annui{label}": f"€{cost_per_year:,.2f}",
            f"Utile operativo medio annuo{label}": f"€{operating_income_per_year:,.2f}",
            f"Utile netto medio annuo{label}": f"€{net_income_per_year:,.2f}",
            f"ROI annuo{label}": f"{roi:.2f} %",
            f"ROS annuo{label}": f"{ros:.2f} %",
            f"ROE annuo{label}": f"{roe:.2f} %",
            f"NPV (5 anni){label}": f"€{npv:,.2f}"
        },
        "generali": {
            "Ticket totali": total_tickets,
            "Ticket per profilo": tickets_per_profile,
            "Tempo chiusura ticket (media/min/max)": f"{avg_time:.2f} / {min_time:.2f} / {max_time:.2f} min",
            "Costo per ticket (media/min/max)": f"€{avg_cost:.2f} / €{min_cost:.2f} / €{max_cost:.2f}",
            "Errori medi per ticket": f"{avg_errors:.2f}",
            "Ticket in ritardo (%)": f"{late_pct:.1f}",
            "Documenti consultati medi": f"{avg_docs:.2f}",
            "Parole medie risposta": f"{avg_words:.2f}",
            "Stress medio": f"{avg_stress:.2f}",
            "Tempo medio scrittura risposta": f"{avg_write_time:.2f} min"
        },
        "documenti": {
            "Documenti totali": len(df_docs),
            "Complessità media": f"{avg_doc_complexity:.2f}",
            "Pagine medie": f"{avg_doc_pages:.2f}",
            "Parole medie": f"{avg_doc_words:.2f}",
            "Tempo medio lettura": f"{avg_doc_read:.2f} min",
            "Tempo medio apertura": f"{avg_doc_open:.2f} min",
            "Tempo medio navigazione": f"{avg_doc_nav:.2f} min",
            "Tempo medio attesa": f"{avg_doc_wait:.2f} min",
            "Tempo medio processing": f"{avg_doc_proc:.2f} min",
            "Errori medi per documento": f"{avg_doc_errors:.2f}",
            "% memory overload": f"{pct_doc_memover:.1f}%",
            "Fonti documenti": doc_sources
        }
    }

def pretty_stats():
    stats_human = compute_statistics_dict(DF_HUMAN, DATA_HUMAN, "")
    stats_ai = compute_statistics_dict(DF_AI, DATA_AI, " (AI)")

    # Accoppia le Statistiche Generali per chiave base
    generali_pairs = []
    for k in stats_human['generali']:
        v_human = stats_human['generali'].get(k, "")
        v_ai = stats_ai['generali'].get(k, "")
        generali_pairs.append((k, v_human, v_ai))

    # Accoppia i KPI per chiave base
    kpi_pairs = []
    for k in stats_human['kpi']:
        base_k = k.replace(" (AI)", "")
        k_ai = base_k + " (AI)"
        v_human = stats_human['kpi'].get(base_k, "")
        v_ai = stats_ai['kpi'].get(k_ai, "")
        kpi_pairs.append((base_k, v_human, v_ai))

    # Accoppia le Statistiche Documenti per chiave base
    documenti_pairs = []
    for k in stats_human['documenti']:
        v_human = stats_human['documenti'].get(k, "")
        v_ai = stats_ai['documenti'].get(k, "")
        # Gestione dizionari (es. Fonti documenti)
        if isinstance(v_human, dict) and isinstance(v_ai, dict):
            v_human_str = ", ".join(f"{kk}: {vv:.2f}" for kk, vv in v_human.items())
            v_ai_str = ", ".join(f"{kk}: {vv:.2f}" for kk, vv in v_ai.items())
            documenti_pairs.append((k, v_human_str, v_ai_str))
        else:
            documenti_pairs.append((k, v_human, v_ai))

    html = f"""
    <style>
    .stat-row {{
        display: flex;
        flex-direction: row;
        flex-wrap: nowrap;
        justify-content: flex-start;
        gap: 18px;
        overflow-x: auto;
    }}
    .stat-card {{
        background: #222; /* più simile a plotly_dark */
        color: #eee;
        border-radius: 12px;
        box-shadow: 0 2px 8px #0002;
        border: 1.5px solid #e5e7eb;
        padding: 18px 22px 12px 22px;
        margin: 8px 0;
        min-width: 270px;
        display: flex;
        flex-direction: column;
        vertical-align: top;
    }}
    .stat-title {{
        font-size: 1.15em;
        font-weight: bold;
        color: #ffd600;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }}
    .stat-list {{
        font-size: 0.8em;
        color: #eee;
        margin: 0;
        padding: 0;
        list-style: none;
    }}
    .stat-list li {{
        margin-bottom: 7px;
    }}
    .stat-kpi-ai {{
        color: #ffd600;
        font-weight: bold;
    }}
    </style>
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-title">Statistiche Generali</div>
            <ul class="stat-list">
                {''.join(
                    f'<li><b>{k}:</b> {v_human} | <span style="color:#ffd600;font-weight:bold;">(AI) {v_ai}</span></li>'
                    for k, v_human, v_ai in generali_pairs
                )}
            </ul>
        </div>
        <div class="stat-card">
            <div class="stat-title">KPI Economici</div>
            <ul class="stat-list">
                {''.join(
                    f'<li><b>{k}:</b> {v_human} | <span style="color:#ffd600;font-weight:bold;">(AI) {v_ai}</span></li>'
                    for k, v_human, v_ai in kpi_pairs
                )}
            </ul>
        </div>
        <div class="stat-card">
            <div class="stat-title">Statistiche Documenti</div>
            <ul class="stat-list">
                {''.join(
                    f'<li><b>{k}:</b> {v_human} | <span style="color:#ffd600;font-weight:bold;">(AI) {v_ai}</span></li>'
                    for k, v_human, v_ai in documenti_pairs
                )}
            </ul>
        </div>
    </div>
    """
    return html

with gr.Blocks() as demo:
    gr.HTML(pretty_stats())

    gr.Markdown("## Distribuzione tempo di chiusura per ticket per profilo")
    with gr.Row():
        out_box_human = gr.Plot(label="Tempo per profilo (Human)")
        out_box_ai = gr.Plot(label="Tempo per profilo (AI)")
        demo.load(boxplot_time_vs_complexity_per_profile, inputs=None, outputs=out_box_human)
        demo.load(boxplot_time_vs_complexity_per_profile_ai, inputs=None, outputs=out_box_ai)

    gr.Markdown("## Distribuzione costo totale per ticket per profilo")
    with gr.Row():
        out_cost_human = gr.Plot(label="Costo per profilo (Human)")
        out_cost_ai = gr.Plot(label="Costo per profilo (AI)")
        demo.load(plot_cost_per_profile, inputs=None, outputs=out_cost_human)
        demo.load(plot_cost_per_profile_ai, inputs=None, outputs=out_cost_ai)

    gr.Markdown("## Boxplot Costo totale per ticket per profilo e fascia di difficoltà media documenti")
    with gr.Row():
        out_cost_complexity_human = gr.Plot(label="Boxplot Costo vs Difficoltà media (Human)")
        out_cost_complexity_ai = gr.Plot(label="Boxplot Costo vs Difficoltà media (AI)")
        demo.load(boxplot_cost_vs_complexity_per_profile, inputs=None, outputs=out_cost_complexity_human)
        demo.load(boxplot_cost_vs_complexity_per_profile_ai, inputs=None, outputs=out_cost_complexity_ai)

if __name__ == "__main__":
    demo.launch()
