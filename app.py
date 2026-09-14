import streamlit as st
from gensim.models import KeyedVectors

st.set_page_config(page_title="Explorador Word2Vec Català", page_icon="🔤")

st.title("🔤 Explorador de Word2Vec en Català")
st.write("Model entrenat amb un corpus en català (CATalog, Projecte AINA) — Treball de Recerca de Denís Bratun")

@st.cache_resource
def load_vectors():
    return KeyedVectors.load("word2vec_catalan_v2.model.wv.vectors.npy", mmap='r')

wv = load_vectors()

st.caption(f"Vocabulari: {len(wv.key_to_index):,} paraules úniques")

tab1, tab2, tab3, tab4 = st.tabs([
    "Paraules similars",
    "Similitud del cosinus",
    "Analogies vectorials",
    "Informació d'una paraula"
])

# --- TAB 1: Top 5 paraules més similars ---
with tab1:
    st.header("Paraules més similars")
    word = st.text_input("Escriu una paraula:", "rei", key="tab1_word")
    topn = st.slider("Nombre de resultats", 3, 20, 5)

    if word:
        word = word.lower().strip()
        try:
            results = wv.most_similar(word, topn=topn)
            for w, score in results:
                st.write(f"**{w}** — similitud: {score:.4f}")
        except KeyError:
            st.error(f"La paraula '{word}' no és al vocabulari del model.")

# --- TAB 2: Similitud del cosinus entre dos paraules ---
with tab2:
    st.header("Similitud del cosinus")
    col1, col2 = st.columns(2)
    with col1:
        w1 = st.text_input("Primera paraula:", "rei", key="tab2_w1")
    with col2:
        w2 = st.text_input("Segona paraula:", "reina", key="tab2_w2")

    if st.button("Calcular similitud"):
        w1_clean = w1.lower().strip()
        w2_clean = w2.lower().strip()
        try:
            sim = wv.similarity(w1_clean, w2_clean)
            st.metric(label=f"Similitud entre '{w1_clean}' i '{w2_clean}'", value=f"{sim:.4f}")
            if sim > 0.6:
                st.success("Similitud alta — paraules molt properes semànticament.")
            elif sim > 0.3:
                st.info("Similitud moderada.")
            else:
                st.warning("Similitud baixa — paraules poc relacionades en aquest model.")
        except KeyError as e:
            st.error(f"Alguna paraula no està al vocabulari: {e}")

# --- TAB 3: Analogies / operacions vectorials ---
with tab3:
    st.header("Analogies vectorials")
    st.caption("Exemple: rei − home + dona = ?")
    col1, col2, col3 = st.columns(3)
    with col1:
        pos1 = st.text_input("Paraula positiva 1", "rei", key="tab3_pos1")
    with col2:
        neg1 = st.text_input("Paraula negativa", "home", key="tab3_neg1")
    with col3:
        pos2 = st.text_input("Paraula positiva 2", "dona", key="tab3_pos2")

    if st.button("Calcular analogia"):
        p1 = pos1.lower().strip()
        n1 = neg1.lower().strip()
        p2 = pos2.lower().strip()
        try:
            result = wv.most_similar(positive=[p1, p2], negative=[n1], topn=5)
            st.write(f"**{p1} − {n1} + {p2} =**")
            for w, score in result:
                st.write(f"  {w}: {score:.4f}")
        except KeyError as e:
            st.error(f"Alguna paraula no està al vocabulari: {e}")

# --- TAB 4: Informació d'una paraula ---
with tab4:
    st.header("Informació d'una paraula")
    w = st.text_input("Escriu una paraula:", "rei", key="tab4_word")

    if w:
        w_clean = w.lower().strip()
        if w_clean in wv.key_to_index:
            idx = wv.key_to_index[w_clean]
            vector = wv[w_clean]
            st.write(f"**Posició al vocabulari:** #{idx + 1}")
            st.write(f"**Dimensions del vector:** {len(vector)}")
            with st.expander("Veure els primers 10 valors del vector"):
                st.write(vector[:10])
        else:
            st.error(f"La paraula '{w}' no és al vocabulari del model.")

st.divider()
st.caption("Treball de Recerca: Les matemàtiques darrere la incrustació de mots — Institut Santa Eulàlia, 2025-2026")