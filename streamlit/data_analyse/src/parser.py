"""
Parser module for reading and analyzing export data.
"""
import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import pandas as pd


@dataclass
class ExportFile:
    """Represents a single exported file."""
    path: Path
    entity_type: str
    record_id: str
    field_type: str
    extension: str
    size: int
    content: Optional[str] = None


@dataclass
class ExportStats:
    """Statistics for an export."""
    total_files: int = 0
    total_size: int = 0
    unique_ids: int = 0
    by_entity: Dict[str, Dict] = field(default_factory=dict)
    by_file_type: Dict[str, int] = field(default_factory=dict)
    year_mentions: Counter = field(default_factory=Counter)
    keyword_counts: Counter = field(default_factory=Counter)
    euro_mentions: List[float] = field(default_factory=list)
    warnings: List[Tuple[str, str, str]] = field(default_factory=list)


def clean_rtf(text: str) -> str:
    """Remove RTF formatting from text."""
    # Remove RTF control words
    text = re.sub(r'\\[a-z0-9]+\s?', ' ', text)
    # Remove braces and backslashes
    text = re.sub(r'[{}\\]', '', text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse export filename to extract ID, field type, and extension.

    Format: {ID}.{FIELD_TYPE}.{extension}
    Example: 100002.GC_INFORMATIE.txt -> (100002, GC_INFORMATIE, txt)
    """
    parts = filename.split('.')
    if len(parts) >= 3:
        record_id = parts[0]
        field_type = '.'.join(parts[1:-1])
        extension = parts[-1]
    elif len(parts) == 2:
        record_id = parts[0]
        field_type = "UNKNOWN"
        extension = parts[1]
    else:
        record_id = filename
        field_type = "UNKNOWN"
        extension = ""

    return record_id, field_type, extension


def scan_export_folder(export_path: Path) -> Tuple[List[ExportFile], ExportStats]:
    """
    Scan an export folder and return all files with statistics.
    """
    files = []
    stats = ExportStats()
    ids_by_entity = defaultdict(set)

    if not export_path.exists():
        return files, stats

    for entity_folder in export_path.iterdir():
        if not entity_folder.is_dir() or not entity_folder.name.startswith('AT_'):
            continue

        entity_type = entity_folder.name
        entity_stats = {'count': 0, 'size': 0, 'types': Counter()}

        for file_path in entity_folder.iterdir():
            if not file_path.is_file():
                continue

            record_id, field_type, extension = parse_filename(file_path.name)
            size = file_path.stat().st_size

            export_file = ExportFile(
                path=file_path,
                entity_type=entity_type,
                record_id=record_id,
                field_type=field_type,
                extension=extension,
                size=size,
            )
            files.append(export_file)

            # Update stats
            stats.total_files += 1
            stats.total_size += size
            ids_by_entity[entity_type].add(record_id)
            entity_stats['count'] += 1
            entity_stats['size'] += size
            entity_stats['types'][f"{field_type}.{extension}"] += 1
            stats.by_file_type[f"{field_type}.{extension}"] += 1

        entity_stats['unique_ids'] = len(ids_by_entity[entity_type])
        stats.by_entity[entity_type] = entity_stats

    stats.unique_ids = sum(len(ids) for ids in ids_by_entity.values())

    return files, stats


def analyze_text_content(
    files: List[ExportFile],
    keywords: List[str],
    max_files: int = 5000
) -> ExportStats:
    """
    Analyze text content of files for patterns.
    """
    stats = ExportStats()

    year_pattern = re.compile(r'\b(19\d{2}|20\d{2})\b')
    euro_pattern = re.compile(r'(\d+[.,]?\d*)\s*euro', re.IGNORECASE)

    txt_files = [f for f in files if f.extension.lower() in ('txt', 'bin')][:max_files]

    for export_file in txt_files:
        try:
            content = export_file.path.read_text(encoding='utf-8', errors='ignore')
            export_file.content = content
            content_lower = clean_rtf(content.lower())

            # Find years
            years = year_pattern.findall(content)
            for year in years:
                stats.year_mentions[year] += 1

            # Find euro amounts
            euros = euro_pattern.findall(content_lower)
            for euro in euros:
                try:
                    amount = float(euro.replace(',', '.'))
                    if 0 < amount < 1000000:  # Filter outliers
                        stats.euro_mentions.append(amount)
                except ValueError:
                    pass

            # Count keywords
            for kw in keywords:
                if kw in content_lower:
                    stats.keyword_counts[kw] += 1

            # Check for warnings (MELDING files or specific phrases)
            if 'MELDING' in export_file.field_type or 'niet welkom' in content_lower or 'niet meer welkom' in content_lower:
                if content.strip():
                    stats.warnings.append((
                        export_file.record_id,
                        export_file.entity_type,
                        clean_rtf(content)[:300]
                    ))

        except Exception:
            pass

    return stats


def get_files_dataframe(files: List[ExportFile]) -> pd.DataFrame:
    """Convert file list to pandas DataFrame."""
    if not files:
        return pd.DataFrame()

    data = []
    for f in files:
        data.append({
            'entity_type': f.entity_type,
            'record_id': f.record_id,
            'field_type': f.field_type,
            'extension': f.extension,
            'size': f.size,
            'path': str(f.path),
        })

    return pd.DataFrame(data)


def get_warnings_dataframe(stats: ExportStats) -> pd.DataFrame:
    """Get warnings as DataFrame."""
    if not stats.warnings:
        return pd.DataFrame(columns=['record_id', 'entity_type', 'melding'])

    return pd.DataFrame(stats.warnings, columns=['record_id', 'entity_type', 'melding'])


def search_content(
    files: List[ExportFile],
    search_term: str,
    max_results: int = 100
) -> List[Tuple[ExportFile, str]]:
    """
    Search for a term in file contents.
    Returns list of (file, context) tuples.
    """
    results = []
    search_lower = search_term.lower()

    txt_files = [f for f in files if f.extension.lower() in ('txt', 'bin')]

    for export_file in txt_files:
        try:
            if export_file.content is None:
                export_file.content = export_file.path.read_text(encoding='utf-8', errors='ignore')

            content_lower = export_file.content.lower()

            if search_lower in content_lower:
                # Find context around match
                idx = content_lower.find(search_lower)
                start = max(0, idx - 50)
                end = min(len(export_file.content), idx + len(search_term) + 50)
                context = clean_rtf(export_file.content[start:end])

                results.append((export_file, f"...{context}..."))

                if len(results) >= max_results:
                    break

        except Exception:
            pass

    return results
