"""
Data Analyse Dashboard - Main Application
==========================================
Streamlit app voor het analyseren van klantdata exports uit Syntess.

Notifica - Business Intelligence voor installatiebedrijven

Analyseert export mappen met structuur:
- AT_RELATIE/  - Klant/leverancier informatie
- AT_DOCUMENT/ - Document notities
- AT_WERK/     - Werkorder notities
- AT_GEBOUW/   - Locatie informatie
- AT_PERSOON/  - Persoonsgegevens
- AT_APPARAAT/ - Apparaat informatie
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from collections import Counter
import zipfile
import tempfile
import shutil
import os

# Local imports
from config import AppConfig, COLORS, ENTITY_TYPES, ANALYSIS_KEYWORDS
from src.parser import (
    scan_export_folder,
    analyze_text_content,
    get_files_dataframe,
    get_warnings_dataframe,
    search_content,
    ExportStats,
)

# Page configuration
st.set_page_config(
    page_title="Data Analyse",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #16136F;
    }
    .warning-card {
        background-color: #fff3cd;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #f39c12;
        margin-bottom: 0.5rem;
    }
    .entity-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .search-result {
        background-color: #e8f4f8;
        border-radius: 4px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .upload-zone {
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f9fafb;
    }
</style>
""", unsafe_allow_html=True)


def get_available_exports(base_path: str) -> list:
    """Scan for available export folders (customer codes)."""
    base = Path(base_path)
    if not base.exists():
        return []

    exports = []
    for folder in base.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            # Check if it contains AT_ folders
            has_at_folders = any(
                f.is_dir() and f.name.startswith('AT_')
                for f in folder.iterdir()
            )
            if has_at_folders:
                exports.append(folder.name)

    return sorted(exports)


def extract_zip_to_temp(uploaded_file) -> tuple:
    """Extract uploaded ZIP file to temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="data_analyse_")

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find the export folder (could be root or nested)
    temp_path = Path(temp_dir)

    # Check if AT_ folders are at root level
    has_at_root = any(f.is_dir() and f.name.startswith('AT_') for f in temp_path.iterdir())
    if has_at_root:
        return temp_path, uploaded_file.name.replace('.zip', '')

    # Check nested folders
    for subfolder in temp_path.iterdir():
        if subfolder.is_dir():
            has_at = any(f.is_dir() and f.name.startswith('AT_') for f in subfolder.iterdir())
            if has_at:
                return subfolder, subfolder.name

    return temp_path, "upload"


def render_sidebar() -> tuple:
    """Render sidebar with export selection."""
    st.sidebar.header("🔍 Data Analyse")

    config = AppConfig()

    # Source selection
    source_mode = st.sidebar.radio(
        "Data bron",
        ["📤 Upload ZIP", "📁 Lokaal pad"],
        index=0,
        help="Upload een ZIP met de export of selecteer een lokaal pad"
    )

    if source_mode == "📤 Upload ZIP":
        st.sidebar.markdown("---")
        uploaded_file = st.sidebar.file_uploader(
            "Upload export ZIP",
            type=['zip'],
            help="Upload een ZIP bestand met de klant export (AT_RELATIE, AT_DOCUMENT, etc.)"
        )

        if uploaded_file:
            # Store in session state
            if 'uploaded_zip' not in st.session_state or st.session_state.uploaded_zip_name != uploaded_file.name:
                with st.spinner("ZIP uitpakken..."):
                    # Clean up previous temp dir
                    if 'temp_dir' in st.session_state and st.session_state.temp_dir:
                        try:
                            shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)
                        except:
                            pass

                    export_path, customer_code = extract_zip_to_temp(uploaded_file)
                    st.session_state.uploaded_zip = True
                    st.session_state.uploaded_zip_name = uploaded_file.name
                    st.session_state.export_path = export_path
                    st.session_state.customer_code = customer_code
                    st.session_state.temp_dir = str(export_path.parent) if export_path.name != "upload" else str(export_path)

            st.sidebar.success(f"✓ {st.session_state.uploaded_zip_name}")
            return st.session_state.export_path, st.session_state.customer_code

        return None, None

    else:
        # Local path mode
        st.sidebar.markdown("---")
        use_custom_path = st.sidebar.checkbox("Aangepast pad", value=False)

        if use_custom_path:
            base_path = st.sidebar.text_input(
                "Export basis pad",
                value=config.default_export_path,
            )
        else:
            base_path = config.default_export_path

        # Get available exports
        available = get_available_exports(base_path)

        if not available:
            st.sidebar.warning("Geen exports gevonden")
            return None, None

        st.sidebar.success(f"{len(available)} exports gevonden")

        # Select export
        selected = st.sidebar.selectbox(
            "Selecteer klant export",
            options=[""] + available,
            format_func=lambda x: "-- Kies een export --" if x == "" else f"Klant {x}",
        )

        if selected:
            export_path = Path(base_path) / selected
            st.sidebar.info(f"📁 {export_path}")
            return export_path, selected

        return None, None


@st.cache_data(ttl=300)
def load_export_data(export_path: str) -> tuple:
    """Load and analyze export data with caching."""
    path = Path(export_path)
    files, file_stats = scan_export_folder(path)
    content_stats = analyze_text_content(files, ANALYSIS_KEYWORDS)

    return files, file_stats, content_stats


def render_kpi_cards(file_stats: ExportStats, content_stats: ExportStats):
    """Render KPI overview cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Totaal Bestanden",
            value=f"{file_stats.total_files:,}",
            help="Aantal bestanden in de export"
        )

    with col2:
        size_kb = file_stats.total_size / 1024
        if size_kb > 1024:
            size_str = f"{size_kb/1024:.1f} MB"
        else:
            size_str = f"{size_kb:.1f} KB"
        st.metric(
            label="Totale Grootte",
            value=size_str,
            help="Totale bestandsgrootte"
        )

    with col3:
        st.metric(
            label="Unieke Records",
            value=f"{file_stats.unique_ids:,}",
            help="Aantal unieke record-IDs"
        )

    with col4:
        warning_count = len(content_stats.warnings)
        st.metric(
            label="Meldingen",
            value=warning_count,
            delta="Let op!" if warning_count > 0 else None,
            delta_color="inverse" if warning_count > 0 else "off",
            help="Aantal waarschuwingen/meldingen"
        )

    with col5:
        euro_count = len(content_stats.euro_mentions)
        st.metric(
            label="Financiële Vermeldingen",
            value=f"{euro_count}",
            help="Aantal keer dat bedragen in euro worden genoemd"
        )


def render_entity_overview(file_stats: ExportStats):
    """Render overview per entity type."""
    st.subheader("Overzicht per Entiteit")

    if not file_stats.by_entity:
        st.info("Geen data beschikbaar")
        return

    # Create DataFrame for display
    data = []
    for entity, stats in file_stats.by_entity.items():
        entity_config = ENTITY_TYPES.get(entity, {})
        data.append({
            'Entiteit': f"{entity_config.get('icon', '📁')} {entity_config.get('label', entity)}",
            'entity_key': entity,
            'Bestanden': stats['count'],
            'Unieke IDs': stats['unique_ids'],
            'Grootte (KB)': round(stats['size'] / 1024, 1),
            'Beschrijving': entity_config.get('description', ''),
        })

    df = pd.DataFrame(data)

    # Bar chart
    fig = px.bar(
        df,
        x='Entiteit',
        y='Bestanden',
        color='Grootte (KB)',
        color_continuous_scale=['#e8f4f8', COLORS['primary']],
        text='Bestanden',
    )
    fig.update_layout(
        height=350,
        showlegend=False,
        xaxis_title="",
        yaxis_title="Aantal bestanden",
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.dataframe(
        df[['Entiteit', 'Bestanden', 'Unieke IDs', 'Grootte (KB)', 'Beschrijving']],
        hide_index=True,
        use_container_width=True,
    )


def render_temporal_analysis(content_stats: ExportStats):
    """Render temporal analysis of year mentions."""
    st.subheader("Temporele Analyse")

    if not content_stats.year_mentions:
        st.info("Geen jaar-vermeldingen gevonden")
        return

    # Filter realistic years
    years = {y: c for y, c in content_stats.year_mentions.items()
             if 1980 <= int(y) <= 2030}

    if not years:
        st.info("Geen relevante jaren gevonden")
        return

    df = pd.DataFrame([
        {'Jaar': int(y), 'Vermeldingen': c}
        for y, c in sorted(years.items())
    ])

    fig = px.bar(
        df,
        x='Jaar',
        y='Vermeldingen',
        color='Vermeldingen',
        color_continuous_scale=['#e8f4f8', COLORS['primary']],
    )
    fig.update_layout(
        height=300,
        showlegend=False,
        xaxis_title="Jaar",
        yaxis_title="Aantal vermeldingen",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Key insights
    if years:
        peak_year = max(years, key=years.get)
        oldest = min(years.keys())
        newest = max(years.keys())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Oudste vermelding", oldest)
        with col2:
            st.metric("Nieuwste vermelding", newest)
        with col3:
            st.metric("Piekjaar", f"{peak_year} ({years[peak_year]}x)")


def render_keyword_analysis(content_stats: ExportStats):
    """Render keyword frequency analysis."""
    st.subheader("Onderwerp Analyse")

    if not content_stats.keyword_counts:
        st.info("Geen keywords gevonden")
        return

    # Create DataFrame
    df = pd.DataFrame([
        {'Onderwerp': kw.capitalize(), 'Aantal': count}
        for kw, count in content_stats.keyword_counts.most_common(15)
    ])

    # Horizontal bar chart
    fig = px.bar(
        df,
        x='Aantal',
        y='Onderwerp',
        orientation='h',
        color='Aantal',
        color_continuous_scale=['#e8f4f8', COLORS['primary']],
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Aantal vermeldingen",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_financial_analysis(content_stats: ExportStats):
    """Render financial mentions analysis."""
    st.subheader("Financiële Analyse")

    if not content_stats.euro_mentions:
        st.info("Geen euro-bedragen gevonden")
        return

    amounts = content_stats.euro_mentions

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Aantal vermeldingen", len(amounts))

    with col2:
        st.metric("Laagste bedrag", f"€ {min(amounts):,.2f}")

    with col3:
        st.metric("Hoogste bedrag", f"€ {max(amounts):,.2f}")

    with col4:
        avg = sum(amounts) / len(amounts)
        st.metric("Gemiddeld bedrag", f"€ {avg:,.2f}")

    # Distribution histogram
    df = pd.DataFrame({'Bedrag': amounts})
    fig = px.histogram(
        df,
        x='Bedrag',
        nbins=30,
        color_discrete_sequence=[COLORS['primary']],
    )
    fig.update_layout(
        height=250,
        xaxis_title="Bedrag (€)",
        yaxis_title="Frequentie",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_warnings_tab(content_stats: ExportStats):
    """Render warnings/alerts tab."""
    st.subheader("Meldingen & Waarschuwingen")

    warnings_df = get_warnings_dataframe(content_stats)

    if warnings_df.empty:
        st.success("Geen meldingen of waarschuwingen gevonden")
        return

    st.warning(f"**{len(warnings_df)} meldingen gevonden** - Deze vereisen mogelijk actie")

    # Filter options
    col1, col2 = st.columns([1, 3])
    with col1:
        entity_filter = st.selectbox(
            "Filter op entiteit",
            options=["Alle"] + list(warnings_df['entity_type'].unique()),
        )

    # Apply filter
    if entity_filter != "Alle":
        warnings_df = warnings_df[warnings_df['entity_type'] == entity_filter]

    # Search in warnings
    with col2:
        search = st.text_input("Zoek in meldingen", placeholder="bijv. 'niet welkom'")

    if search:
        warnings_df = warnings_df[
            warnings_df['melding'].str.lower().str.contains(search.lower(), na=False)
        ]

    # Display warnings
    st.markdown(f"**{len(warnings_df)} resultaten**")

    for _, row in warnings_df.head(50).iterrows():
        entity_config = ENTITY_TYPES.get(row['entity_type'], {})
        icon = entity_config.get('icon', '📁')

        st.markdown(f"""
        <div class="warning-card">
            <strong>{icon} {row['entity_type']} - ID: {row['record_id']}</strong><br>
            {row['melding']}
        </div>
        """, unsafe_allow_html=True)

    if len(warnings_df) > 50:
        st.info(f"Nog {len(warnings_df) - 50} meldingen niet getoond...")

    # Export button
    if not warnings_df.empty:
        csv = warnings_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download meldingen als CSV",
            csv,
            "meldingen_export.csv",
            "text/csv",
        )


def render_search_tab(files: list):
    """Render search tab."""
    st.subheader("Zoeken in Content")

    search_term = st.text_input(
        "Zoekterm",
        placeholder="bijv. 'korting', 'ordernummer', 'contract'",
        help="Zoek in alle tekstbestanden"
    )

    if search_term and len(search_term) >= 2:
        with st.spinner(f"Zoeken naar '{search_term}'..."):
            results = search_content(files, search_term, max_results=100)

        if results:
            st.success(f"**{len(results)} resultaten** gevonden")

            for export_file, context in results:
                entity_config = ENTITY_TYPES.get(export_file.entity_type, {})
                icon = entity_config.get('icon', '📁')

                st.markdown(f"""
                <div class="search-result">
                    <strong>{icon} {export_file.entity_type}</strong> |
                    ID: {export_file.record_id} |
                    {export_file.field_type}<br>
                    {context}
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info(f"Geen resultaten gevonden voor '{search_term}'")

    else:
        st.info("Voer een zoekterm in (minimaal 2 karakters)")


def render_export_tab(files: list, file_stats: ExportStats, content_stats: ExportStats, customer_code: str):
    """Render data export tab."""
    st.subheader("Data Export")

    st.markdown("Exporteer de geanalyseerde data naar verschillende formaten.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bestandslijst**")
        files_df = get_files_dataframe(files)
        if not files_df.empty:
            csv = files_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download bestandslijst (CSV)",
                csv,
                f"bestanden_{customer_code}.csv",
                "text/csv",
            )
            st.caption(f"{len(files_df)} bestanden")

    with col2:
        st.markdown("**Meldingen**")
        warnings_df = get_warnings_dataframe(content_stats)
        if not warnings_df.empty:
            csv = warnings_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download meldingen (CSV)",
                csv,
                f"meldingen_{customer_code}.csv",
                "text/csv",
            )
            st.caption(f"{len(warnings_df)} meldingen")
        else:
            st.caption("Geen meldingen")

    # Summary report
    st.markdown("---")
    st.markdown("**Samenvatting Rapport**")

    report = f"""# Data Analyse Rapport - Klant {customer_code}

Gegenereerd: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Totaal Overzicht
- Totaal bestanden: {file_stats.total_files:,}
- Totale grootte: {file_stats.total_size / 1024:.1f} KB
- Unieke records: {file_stats.unique_ids:,}
- Meldingen/waarschuwingen: {len(content_stats.warnings)}

## Per Entiteit
"""
    for entity, stats in file_stats.by_entity.items():
        entity_label = ENTITY_TYPES.get(entity, {}).get('label', entity)
        report += f"- **{entity_label}**: {stats['count']} bestanden, {stats['unique_ids']} unieke IDs\n"

    report += f"""
## Top Onderwerpen
"""
    for kw, count in content_stats.keyword_counts.most_common(10):
        report += f"- {kw.capitalize()}: {count}x\n"

    if content_stats.warnings:
        report += f"""
## Waarschuwingen ({len(content_stats.warnings)})
"""
        for record_id, entity, text in content_stats.warnings[:20]:
            report += f"- [{entity}:{record_id}] {text[:100]}...\n"

    st.download_button(
        "📥 Download rapport (Markdown)",
        report,
        f"rapport_{customer_code}.md",
        "text/markdown",
    )


def main():
    """Main application entry point."""
    st.title("🔍 Data Analyse")
    st.caption("Analyse van klantdata exports | Notifica BI")

    # Sidebar - export selection
    export_path, customer_code = render_sidebar()

    if not export_path:
        st.markdown("""
        <div class="upload-zone">
            <h3>👈 Upload een export of selecteer een lokaal pad</h3>
            <p style="color: #6b7280;">
                Upload een ZIP bestand met de klant export uit Syntess,<br>
                of selecteer een lokaal pad als je de app lokaal draait.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        ### Wat doet deze tool?

        Deze tool analyseert data exports uit Syntess en geeft inzicht in:

        - **Overzicht** - Aantal bestanden, grootte, unieke records per entiteit
        - **Temporele analyse** - Welke jaren worden genoemd in de notities
        - **Onderwerp analyse** - Meest voorkomende onderwerpen (contract, onderhoud, etc.)
        - **Meldingen** - Waarschuwingen en kritieke informatie
        - **Zoeken** - Doorzoek alle content op specifieke termen
        - **Export** - Download analyses als CSV of rapport

        ### Verwachte mapstructuur (in ZIP of folder)

        ```
        {klantnummer}/
        ├── AT_RELATIE/
        │   ├── 100001.GC_INFORMATIE.txt
        │   └── 100001.MELDING.txt
        ├── AT_DOCUMENT/
        ├── AT_WERK/
        ├── AT_GEBOUW/
        ├── AT_PERSOON/
        └── AT_APPARAAT/
        ```
        """)
        return

    # Load data
    with st.spinner("Data laden en analyseren..."):
        files, file_stats, content_stats = load_export_data(str(export_path))

    if not files:
        st.error("Geen bestanden gevonden in deze export")
        return

    st.sidebar.markdown(f"*{len(files)} bestanden geladen*")

    # KPI Cards
    st.markdown("---")
    render_kpi_cards(file_stats, content_stats)

    # Tabs
    tab_overview, tab_temporal, tab_keywords, tab_financial, tab_warnings, tab_search, tab_export = st.tabs([
        "📊 Overzicht",
        "📅 Temporeel",
        "🏷️ Onderwerpen",
        "💰 Financieel",
        "⚠️ Meldingen",
        "🔎 Zoeken",
        "📥 Export",
    ])

    with tab_overview:
        render_entity_overview(file_stats)

    with tab_temporal:
        render_temporal_analysis(content_stats)

    with tab_keywords:
        render_keyword_analysis(content_stats)

    with tab_financial:
        render_financial_analysis(content_stats)

    with tab_warnings:
        render_warnings_tab(content_stats)

    with tab_search:
        render_search_tab(files)

    with tab_export:
        render_export_tab(files, file_stats, content_stats, customer_code)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "Data Analyse Dashboard v0.1 | Notifica - Business Intelligence voor installatiebedrijven"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
