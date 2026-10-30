import io
import os
from typing import Any

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

PAGE_TITLE = "Insight Lens"
MODEL_NAME = "gemini-3.8-flash"

ANALYST_SYSTEM = """Você é o Agente Data Analyst do Insight Lens.
Analise a saúde, qualidade e padrões estatísticos de um conjunto de dados tabular.

Com base APENAS no resumo estatístico estruturado fornecido:
1. Avalie a qualidade dos dados (nulos, tipos, cardinalidade, possíveis outliers).
2. Identifique padrões, distribuições, correlações e assimetrias relevantes.
3. Aponte inconsistências, vieses amostrais e limitações da análise.
4. Destaque os achados mais importantes com evidência numérica.

Responda em português, em Markdown, com seções claras e tom técnico, objetivo e preciso.
Não invente colunas, valores ou fatos que não estejam no resumo."""

ADVISOR_SYSTEM = """Você é o Agente Business Advisor do Insight Lens.
Recebe a análise técnica de um Data Analyst e traduz os achados em recomendações de negócio.

Com base na análise técnica e no resumo estatístico:
1. Extraia implicações de negócio concretas (risco, oportunidade, eficiência, receita, custo).
2. Proponha estratégias acionáveis, priorizadas (curto, médio e longo prazo).
3. Sugira KPIs e próximos experimentos ou coletas de dados.
4. Seja específico: ligue cada recomendação a um padrão observado nos dados.

Responda em português, em Markdown, com seções claras e tom executivo.
Não contradiga os dados; se a evidência for fraca, deixe isso explícito."""


def configure_page() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            html, body, [class*="css"] {
                font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;
            }
            .stApp {
                background: linear-gradient(180deg, #f6f7fb 0%, #eef1f7 100%);
            }
            [data-testid="stSidebar"] {
                background: #0f172a;
                border-right: none;
            }
            [data-testid="stSidebar"] * {
                color: #e2e8f0;
            }
            [data-testid="stSidebar"] .stMarkdown p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span {
                color: #cbd5e1 !important;
            }
            [data-testid="stHeader"] {
                background: transparent;
            }
            .hero-kicker {
                letter-spacing: 0.14em;
                text-transform: uppercase;
                font-size: 0.72rem;
                font-weight: 600;
                color: #4f46e5;
                margin-bottom: 0.25rem;
            }
            .hero-title {
                font-size: 2.15rem;
                font-weight: 700;
                color: #0f172a;
                letter-spacing: -0.03em;
                margin: 0;
            }
            .hero-sub {
                color: #64748b;
                font-size: 1.02rem;
                margin-top: 0.4rem;
                margin-bottom: 1.4rem;
            }
            div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 0.85rem 1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
            }
            .section-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                padding: 1.1rem 1.2rem 0.4rem 1.2rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
            }
            .section-title {
                font-weight: 650;
                color: #0f172a;
                font-size: 1.05rem;
                margin-bottom: 0.15rem;
            }
            .stButton > button {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.65rem 1.2rem;
                font-weight: 600;
                width: 100%;
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
                color: white;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.4rem;
            }
            .stTabs [data-baseweb="tab"] {
                background: #ffffff;
                border-radius: 12px;
                padding: 0.55rem 1rem;
                border: 1px solid #e2e8f0;
            }
            .empty-state {
                background: #ffffff;
                border: 1px dashed #cbd5e1;
                border-radius: 18px;
                padding: 2.4rem 1.5rem;
                text-align: center;
                color: #64748b;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_api_key(sidebar_value: str) -> str:
    if sidebar_value.strip():
        return sidebar_value.strip()
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if secret_key:
            return str(secret_key)
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "").strip()


def load_csv(uploaded_file: Any) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(raw))


def build_structured_summary(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame(
        {
            "nulos": missing,
            "percentual": missing_pct,
        }
    )
    missing_table = missing_table[missing_table["nulos"] > 0].sort_values(
        "nulos", ascending=False
    )

    nunique = df.nunique(dropna=True).sort_values(ascending=False)
    numeric_df = df.select_dtypes(include="number")
    describe_numeric = (
        numeric_df.describe().T.to_string() if not numeric_df.empty else "Sem colunas numéricas."
    )

    categorical_df = df.select_dtypes(exclude="number")
    if not categorical_df.empty:
        cat_lines = []
        for col in categorical_df.columns[:20]:
            top = categorical_df[col].value_counts(dropna=True).head(5)
            cat_lines.append(f"- {col}: {top.to_dict()}")
        categorical_preview = "\n".join(cat_lines)
    else:
        categorical_preview = "Sem colunas categóricas/texto."

    correlation = "Não aplicável (menos de 2 colunas numéricas)."
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        correlation = corr.round(3).to_string()

    summary = f"""
# Resumo estatístico estruturado

## Dimensões
- linhas: {df.shape[0]}
- colunas: {df.shape[1]}
- colunas: {list(df.columns)}

## Tipos de dados
{df.dtypes.astype(str).to_string()}

## Info do DataFrame
{info_text}

## Valores ausentes
{missing_table.to_string() if not missing_table.empty else "Nenhum valor nulo detectado."}

## Cardinalidade (valores únicos)
{nunique.to_string()}

## Estatísticas descritivas (numéricas)
{describe_numeric}

## Top categorias (até 5 valores por coluna)
{categorical_preview}

## Correlação entre variáveis numéricas
{correlation}

## Amostra (head)
{df.head(8).to_csv(index=False)}
""".strip()
    return summary


def run_agent(
    client: genai.Client,
    system_instruction: str,
    user_contents: str,
) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("A API do Gemini retornou uma resposta vazia.")
    return text


def render_dataframe_section(title: str, data: pd.DataFrame) -> None:
    st.markdown(
        f'<div class="section-card"><div class="section-title">{title}</div></div>',
        unsafe_allow_html=True,
    )
    st.dataframe(data, use_container_width=True, hide_index=False)


def main() -> None:
    configure_page()

    if "tech_analysis" not in st.session_state:
        st.session_state.tech_analysis = ""
    if "business_insights" not in st.session_state:
        st.session_state.business_insights = ""
    if "last_file_id" not in st.session_state:
        st.session_state.last_file_id = None

    with st.sidebar:
        st.markdown("### 🔎 Insight Lens")
        st.caption("Upload um CSV e obtenha leitura técnica + insights de negócio.")
        st.divider()
        uploaded_file = st.file_uploader(
            "Arquivo CSV",
            type=["csv"],
            help="Selecione um arquivo .csv para análise.",
        )
        api_key_input = st.text_input(
            "Chave da API Gemini",
            type="password",
            placeholder="GEMINI_API_KEY",
            help="Também pode ser definida em variável de ambiente ou em .streamlit/secrets.toml.",
        )
        st.caption("A chave não é enviada para o navegador de terceiros; fica nesta sessão.")

    st.markdown('<p class="hero-kicker">CSV · Pandas · Gemini</p>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="hero-title">{PAGE_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Carregue um CSV para inspecionar os dados e, em seguida, analise com dois agentes de IA em sequência.</p>',
        unsafe_allow_html=True,
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="empty-state">Envie um arquivo <strong>.csv</strong> na barra lateral para começar.</div>',
            unsafe_allow_html=True,
        )
        return

    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.last_file_id != file_id:
        st.session_state.tech_analysis = ""
        st.session_state.business_insights = ""
        st.session_state.last_file_id = file_id

    try:
        df = load_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Não foi possível ler o CSV. Verifique o formato do arquivo. Detalhe: {exc}")
        return

    if df.empty:
        st.warning("O arquivo foi lido, mas está vazio.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Linhas", f"{df.shape[0]:,}".replace(",", "."))
    metric_cols[1].metric("Colunas", df.shape[1])
    metric_cols[2].metric("Nulos", int(df.isna().sum().sum()))
    metric_cols[3].metric("Colunas numéricas", df.select_dtypes(include="number").shape[1])

    st.markdown("### Visão geral dos dados")
    render_dataframe_section("Primeiras linhas (head)", df.head())

    dtype_df = (
        df.dtypes.astype(str)
        .rename("tipo")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "coluna"})
    )
    missing_df = (
        df.isna()
        .sum()
        .rename("nulos")
        .to_frame()
        .assign(percentual=lambda x: (x["nulos"] / len(df) * 100).round(2))
        .reset_index()
        .rename(columns={"index": "coluna"})
    )
    col_left, col_right = st.columns(2)
    with col_left:
        render_dataframe_section("Tipos de dados", dtype_df)
    with col_right:
        render_dataframe_section("Valores ausentes", missing_df)

    numeric_describe = df.describe(include="number").T
    if not numeric_describe.empty:
        render_dataframe_section("Estatísticas descritivas", numeric_describe)
    else:
        st.info("Não há colunas numéricas para estatísticas descritivas.")

    st.divider()
    analyze = st.button("Analisar com IA", type="primary")

    if analyze:
        api_key = get_api_key(api_key_input)
        if not api_key:
            st.error(
                "Informe a chave da API Gemini na barra lateral, em GEMINI_API_KEY ou em .streamlit/secrets.toml."
            )
        else:
            summary = build_structured_summary(df)
            client = genai.Client(api_key=api_key)

            with st.status("Executando agentes em sequência...", expanded=True) as status:
                try:
                    st.write("1/2 · Agente Data Analyst")
                    tech = run_agent(
                        client,
                        ANALYST_SYSTEM,
                        "Analise o seguinte resumo estatístico estruturado:\n\n" + summary,
                    )
                    st.session_state.tech_analysis = tech

                    st.write("2/2 · Agente Business Advisor")
                    business = run_agent(
                        client,
                        ADVISOR_SYSTEM,
                        (
                            "Traduza os achados abaixo em estratégias de negócio.\n\n"
                            "## Análise técnica do Data Analyst\n"
                            f"{tech}\n\n"
                            "## Resumo estatístico original\n"
                            f"{summary}"
                        ),
                    )
                    st.session_state.business_insights = business
                    status.update(label="Análise concluída", state="complete")
                except Exception as exc:
                    status.update(label="Falha na análise com IA", state="error")
                    st.error(f"Não foi possível concluir a análise com o Gemini. Detalhe: {exc}")

    if st.session_state.tech_analysis or st.session_state.business_insights:
        tab_tech, tab_biz = st.tabs(["📊 Análise Técnica", "💡 Insights de Negócio"])
        with tab_tech:
            st.markdown(st.session_state.tech_analysis or "_Ainda sem análise técnica._")
        with tab_biz:
            st.markdown(st.session_state.business_insights or "_Ainda sem insights de negócio._")


if __name__ == "__main__":
    main()
