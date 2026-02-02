"""
Matching algorithms for Taken and WV/Balans mapping.
"""
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple
import re


def token_set_ratio(s1: str, s2: str) -> float:
    """
    Calculate token set similarity ratio (similar to fuzzywuzzy token_set_ratio).
    Order-independent comparison of token sets.
    """
    if not s1 or not s2:
        return 0.0

    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    # Intersection and differences
    intersection = tokens1 & tokens2
    diff1 = tokens1 - tokens2
    diff2 = tokens2 - tokens1

    # Build comparison strings
    sorted_intersection = ' '.join(sorted(intersection))
    combined1 = ' '.join(sorted(intersection | diff1))
    combined2 = ' '.join(sorted(intersection | diff2))

    # Compare different combinations
    ratios = []

    if sorted_intersection:
        ratios.append(SequenceMatcher(None, sorted_intersection, combined1).ratio())
        ratios.append(SequenceMatcher(None, sorted_intersection, combined2).ratio())
        ratios.append(SequenceMatcher(None, combined1, combined2).ratio())
    else:
        ratios.append(SequenceMatcher(None, combined1, combined2).ratio())

    return max(ratios) * 100 if ratios else 0.0


def token_sort_ratio(s1: str, s2: str) -> float:
    """
    Calculate token sort ratio: sort tokens alphabetically and compare.
    """
    if not s1 or not s2:
        return 0.0

    sorted1 = ' '.join(sorted(s1.lower().split()))
    sorted2 = ' '.join(sorted(s2.lower().split()))

    return SequenceMatcher(None, sorted1, sorted2).ratio() * 100


def prefix_match(target: str, candidate: str) -> bool:
    """
    Check if candidate is a prefix of target or vice versa (token-level).
    """
    target_tokens = target.lower().split()
    candidate_tokens = candidate.lower().split()

    if not target_tokens or not candidate_tokens:
        return False

    # Check if one is prefix of the other
    min_len = min(len(target_tokens), len(candidate_tokens))
    return target_tokens[:min_len] == candidate_tokens[:min_len]


def subterm_match(target: str, candidate: str) -> bool:
    """
    Check if one string fully contains the other (token-level).
    """
    target_tokens = set(target.lower().split())
    candidate_tokens = set(candidate.lower().split())

    if not target_tokens or not candidate_tokens:
        return False

    return target_tokens.issubset(candidate_tokens) or candidate_tokens.issubset(target_tokens)


class MappingIndex:
    """
    Index for efficient mapping lookups with frequency-based tie-breaking.
    """

    def __init__(self):
        # Main index: normalized_key -> (code, group/niveau) with frequencies
        self.exact_index: Dict[str, Dict[Tuple, int]] = defaultdict(lambda: defaultdict(int))

        # Token index for fuzzy matching candidate narrowing
        self.token_index: Dict[str, Set[str]] = defaultdict(set)

        # First-word index
        self.first_word_index: Dict[str, Set[str]] = defaultdict(set)

        # Global frequencies
        self.global_freq: Dict[Tuple, int] = defaultdict(int)

        # Type-specific indices (for Taken)
        self.type_index: Dict[str, Dict[str, Dict[Tuple, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # Client-specific indices
        self.client_index: Dict[str, Dict[str, Dict[Tuple, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # Resolved best combinations per key
        self._resolved: Dict[str, Tuple] = {}
        self._resolved_by_type: Dict[str, Dict[str, Tuple]] = defaultdict(dict)

    def add(
        self,
        normalized_key: str,
        combination: Tuple,
        taak_type: Optional[str] = None,
        client_id: Optional[str] = None
    ):
        """Add a mapping entry to the index."""
        if not normalized_key or not combination:
            return

        # Update exact index
        self.exact_index[normalized_key][combination] += 1
        self.global_freq[combination] += 1

        # Update token index
        tokens = normalized_key.split()
        for token in tokens:
            if len(token) >= 3:
                self.token_index[token].add(normalized_key)

        # Update first-word index
        if tokens:
            self.first_word_index[tokens[0]].add(normalized_key)

        # Update type-specific index
        if taak_type:
            type_key = taak_type.strip().lower()
            self.type_index[type_key][normalized_key][combination] += 1

        # Update client-specific index
        if client_id:
            self.client_index[client_id][normalized_key][combination] += 1

    def resolve(self):
        """
        Resolve the best combination for each key using tie-break rules:
        1. Highest frequency within that key
        2. Highest global frequency
        3. Lowest numeric code
        """
        self._resolved = {}

        for key, combinations in self.exact_index.items():
            self._resolved[key] = self._select_best(combinations)

        # Resolve by type
        for type_key, type_data in self.type_index.items():
            for key, combinations in type_data.items():
                self._resolved_by_type[type_key][key] = self._select_best(combinations)

    def _select_best(self, combinations: Dict[Tuple, int]) -> Tuple:
        """Select best combination using tie-break rules."""
        if not combinations:
            return None

        candidates = list(combinations.items())

        # Sort by: -local_freq, -global_freq, numeric_code
        def sort_key(item):
            combo, local_freq = item
            global_freq = self.global_freq.get(combo, 0)

            # Extract numeric code for comparison
            code = combo[0] if combo else ''
            try:
                numeric_code = int(str(code).replace('.0', ''))
            except (ValueError, TypeError):
                numeric_code = float('inf')

            return (-local_freq, -global_freq, numeric_code)

        candidates.sort(key=sort_key)
        return candidates[0][0]

    def get_exact(self, normalized_key: str, taak_type: Optional[str] = None) -> Optional[Tuple]:
        """Get exact match for a normalized key."""
        if taak_type:
            type_key = taak_type.strip().lower()
            if type_key in self._resolved_by_type and normalized_key in self._resolved_by_type[type_key]:
                return self._resolved_by_type[type_key][normalized_key]

        return self._resolved.get(normalized_key)

    def get_candidates(self, normalized_key: str, taak_type: Optional[str] = None) -> Set[str]:
        """Get candidate keys that share tokens with the target."""
        candidates = set()
        tokens = normalized_key.split()

        for token in tokens:
            if len(token) >= 3:
                candidates.update(self.token_index.get(token, set()))

        if tokens:
            candidates.update(self.first_word_index.get(tokens[0], set()))

        # Filter by type if specified
        if taak_type:
            type_key = taak_type.strip().lower()
            if type_key in self._resolved_by_type:
                candidates = candidates & set(self._resolved_by_type[type_key].keys())

        return candidates

    def get_all_keys(self, taak_type: Optional[str] = None) -> Set[str]:
        """Get all keys in the index."""
        if taak_type:
            type_key = taak_type.strip().lower()
            if type_key in self._resolved_by_type:
                return set(self._resolved_by_type[type_key].keys())
        return set(self._resolved.keys())

    def get_top_by_type(self, taak_type: str) -> Optional[Tuple]:
        """Get the most common combination for a given type."""
        type_key = taak_type.strip().lower()

        if type_key not in self.type_index:
            return None

        # Aggregate all combinations for this type
        type_combos: Dict[Tuple, int] = defaultdict(int)
        for key_combos in self.type_index[type_key].values():
            for combo, freq in key_combos.items():
                type_combos[combo] += freq

        if not type_combos:
            return None

        return self._select_best(type_combos)

    def get_top_by_client_and_type(self, client_id: str, taak_type: str) -> Optional[Tuple]:
        """Get the most common combination for a given client and type."""
        if client_id not in self.client_index:
            return None

        type_key = taak_type.strip().lower()

        # Filter client entries by type
        client_type_combos: Dict[Tuple, int] = defaultdict(int)
        for key, combos in self.client_index[client_id].items():
            # Check if this key exists in the type index
            if type_key in self.type_index and key in self.type_index[type_key]:
                for combo, freq in combos.items():
                    client_type_combos[combo] += freq

        if not client_type_combos:
            return None

        return self._select_best(client_type_combos)


class TakenMatcher:
    """
    Matcher for Taken (task) mapping with all specified matching strategies.
    """

    # Anchor keys that get special treatment
    ANCHOR_KEYS = ['verzuim', 'scholing', 'reisuren', 'urenregistratie', 'verlof', 'projectgebonden', 'montage']

    def __init__(self, index: MappingIndex):
        self.index = index
        self.match_stats = Counter()

    def match(
        self,
        normalized_key: str,
        taak_type: str,
        client_id: Optional[str] = None
    ) -> Tuple[Optional[Tuple], str]:
        """
        Match a normalized task key to a mapping combination.

        Args:
            normalized_key: Normalized task key
            taak_type: Type (Direct/Indirect) - required
            client_id: Optional client ID for client-specific matching

        Returns:
            Tuple of (combination, match_method) or (None, 'unmatched')
        """
        type_key = taak_type.strip().lower() if taak_type else ''

        # Step A: Exact match
        result = self.index.get_exact(normalized_key, taak_type)
        if result:
            self.match_stats['A_exact'] += 1
            return result, 'A_exact'

        # Step B: Anchor match
        if normalized_key in self.ANCHOR_KEYS:
            result = self.index.get_exact(normalized_key, taak_type)
            if result:
                self.match_stats['B_anchor'] += 1
                return result, 'B_anchor'

        # Step C: Prefix/subterm match
        candidates = self.index.get_candidates(normalized_key, taak_type)
        for candidate_key in candidates:
            if prefix_match(normalized_key, candidate_key) or subterm_match(normalized_key, candidate_key):
                result = self.index.get_exact(candidate_key, taak_type)
                if result:
                    self.match_stats['C_prefix'] += 1
                    return result, 'C_prefix'

        # Step D: Token set ratio >= 90%
        best_score = 0
        best_result = None
        best_key = None

        for candidate_key in candidates:
            score = token_set_ratio(normalized_key, candidate_key)
            if score >= 90 and score > best_score:
                result = self.index.get_exact(candidate_key, taak_type)
                if result:
                    best_score = score
                    best_result = result
                    best_key = candidate_key

        if best_result:
            self.match_stats['D_token_set'] += 1
            return best_result, 'D_token_set'

        # Step E: Token sort + difflib >= 90%
        best_score = 0
        best_result = None

        for candidate_key in candidates:
            score = token_sort_ratio(normalized_key, candidate_key)
            if score >= 90 and score > best_score:
                result = self.index.get_exact(candidate_key, taak_type)
                if result:
                    best_score = score
                    best_result = result

        if best_result:
            self.match_stats['E_token_sort'] += 1
            return best_result, 'E_token_sort'

        # Step G: Token-based majority voting
        tokens = [t for t in normalized_key.split() if len(t) >= 3]
        if tokens:
            votes: Dict[Tuple, int] = defaultdict(int)

            for token in tokens:
                token_keys = self.index.token_index.get(token, set())
                for key in token_keys:
                    combo = self.index.get_exact(key, taak_type)
                    if combo:
                        votes[combo] += 1

            if votes:
                # Get winner with >= 2 votes
                top_votes = sorted(votes.items(), key=lambda x: (-x[1], x[0]))
                if top_votes[0][1] >= 2:
                    self.match_stats['G_majority'] += 1
                    return top_votes[0][0], 'G_majority'

        # Step H: Backstop
        if client_id:
            result = self.index.get_top_by_client_and_type(client_id, taak_type)
            if result:
                self.match_stats['H_client_top'] += 1
                return result, 'H_client_top'

        result = self.index.get_top_by_type(taak_type)
        if result:
            self.match_stats['H_type_top'] += 1
            return result, 'H_type_top'

        self.match_stats['unmatched'] += 1
        return None, 'unmatched'


class WVBalansMatcher:
    """
    Matcher for WV/Balans (rubriek) mapping with anchors and guardrails.
    """

    # Fixed anchors with exact codes
    FIXED_ANCHORS = [
        (r'^omzet$|^balie omzet$', ('101001', 'Omzet', 'Omzet')),
        (r'^afschrijving$|^afschr$', ('109001', 'Afschrijving', 'Afschrijving')),
    ]

    # Anchors that map to (Niveau1, Niveau2) - code comes from mapping
    NIVEAU_ANCHORS = [
        # === PERSONNEL COSTS ===
        (r'^pensioen', ('Personeelkosten', 'Pensioenlasten')),
        (r'^reiskosten|woon.*werk|reis.*vergoeding|vergoeding.*woon', ('Personeelkosten', 'Personeelskosten Overig')),
        (r'sociale verzekeringswet|svw|\bwia\b|\bwga\b|werkgeversdeel|wg deel|sociaal fonds',
         ('Personeelkosten', 'Sociale lasten')),
        (r'inhouding.*wao|afdracht.*wao|inhouding.*wia|afdracht.*wia|\bwao\b',
         ('Personeelkosten', 'Sociale lasten')),
        (r'heffing.*social|sociale.*heffing|eindheffing|heffing.*regeling',
         ('Personeelkosten', 'Sociale lasten')),
        (r'vakantie.*geld|vakantie.*toesla|mutatie.*reserv.*vakantie|vakantie.*verplichting',
         ('Personeelkosten', 'Lonen en salarissen')),
        (r'loon.*kosten|salaris(?!.*vrijwillig)', ('Personeelkosten', 'Lonen en salarissen')),
        (r'werkkostenregeling|\bwkr\b', ('Personeelkosten', 'Personeelskosten Overig')),
        (r'vergoeding.*directie|directie.*vergoeding|management.*vergoeding',
         ('Personeelkosten', 'Lonen en salarissen')),
        (r'ongevallenverzekering', ('Personeelkosten', 'Personeelskosten Overig')),
        (r'\bvtk\b|verkorte.*werkweek', ('Personeelkosten', 'Indirecte arbeidkosten')),

        # === DEPRECIATION ===
        (r'afschrijving.*kst|afschrijvingskst|afschr.*kst|afschr.*vervoer|afschrijving.*vervoer',
         ('Afschrijving', 'Afschrijving')),

        # === DIRECT COSTS ===
        (r'^inleen', ('Directe kosten', 'Onderaanneming')),
        (r'toeslag onderaanneming', ('Directe kosten', 'Onderaanneming')),
        (r'toeslag (montage|ondersteuning|materiaal)', ('Directe kosten', None)),
        (r'\bprefab\b|prefabkosten', ('Directe kosten', 'Materiaal')),
        (r'directe.*uren|uren.*direct', ('Directe kosten', 'Directe arbeidkosten')),

        # === REVENUE ===
        (r'gefactureerd.*ew|gefactureerde.*termijnen|gefactureerde.*verkopen', ('Omzet', 'Gefactureerde termijnen')),
        (r'opbrengst afgesloten werken|kosten afgesloten werken|waarderingsresultaat|schade',
         ('Omzet', 'Projectwaardering /-resultaat')),
        (r'renovatie|grote.*renovatie|renovatie.*klein', ('Omzet', 'Omzet')),

        # === OTHER BUSINESS COSTS ===
        (r'^onderhoud', ('Overige bedrijfskosten', 'Gereedschaps- en exploitatiekosten')),
        (r'tekenmateriaal', ('Overige bedrijfskosten', 'Kantoorkosten')),
        (r'container|containerhuur|vracht', ('Overige bedrijfskosten', 'Huisvestingkosten')),
        (r'zuurstof|gas', ('Directe kosten', 'Materiaal')),
        (r'voorraad', ('Omzet', 'Projectmutaties')),
        (r'doorbelasting\s+ohw|\bohw\b|doorbelasting\s+onderhanden werk', ('Omzet', 'Projectmutaties')),
        (r'interest|rente', ('Financiele Baten en Lasten', 'Financiele Baten en Lasten')),
        (r'bedrijfsaansprakelijkheidsverzekering', ('Overige bedrijfskosten', 'Verzekeringskosten')),
        (r'datacommunicatie|telecommunicatie|telefoonvergoeding|telefoon|internet',
         ('Overige bedrijfskosten', 'ICT en Communicatiekosten')),
        (r'parkeerkosten|bedrijfsauto|autokosten', ('Overige bedrijfskosten', 'Autokosten en overige Transportkosten')),
        (r'cjib|boete|fiscaal.*niet.*aftrek', ('Overige bedrijfskosten', 'Autokosten en overige Transportkosten')),
        (r'gereedschap', ('Overige bedrijfskosten', 'Gereedschaps- en exploitatiekosten')),
        (r'automatiseringskosten|ict.*kosten', ('Overige bedrijfskosten', 'ICT en Communicatiekosten')),
        (r'alarm|alarmopvolging|beveiliging', ('Overige bedrijfskosten', 'ICT en Communicatiekosten')),
        (r'receptie|secretariaat', ('Overige bedrijfskosten', 'Algemene kosten')),
        (r'certificat|erkenning|keurmerk', ('Overige bedrijfskosten', 'Algemene kosten')),
        (r'mutatie.*voorziening', ('Financiele Baten en Lasten', 'Overige bijzondere baten en lasten')),
        (r'opslag doorbelast|doorberekening', ('Overige bedrijfskosten', 'Doorbelastingen')),
        # Note: Generic ^resultaat moved to AFTER Balans patterns to allow Balans-specific matching first

        # === BALANS - VLOTTENDE ACTIVA ===
        (r'voorraad|magazijn|goederen', ('Vlottende Activa', 'Voorraden')),
        (r'debiteuren|debiteur|te ontvangen', ('Vlottende Activa', 'Debiteuren')),
        (r'onderhanden.*project|ohw|onderhanden.*werk', ('Vlottende Activa', 'Onderhanden projecten')),
        (r'bank|rekening.*courant|spaarrekening|g.*rekening', ('Vlottende Activa', 'Liquide Middelen')),
        (r'\bkas\b|kasgeld|contant', ('Vlottende Activa', 'Liquide Middelen')),
        (r'liquide.*middelen', ('Vlottende Activa', 'Liquide Middelen')),
        (r'vordering.*groep|intercompany.*vordering|r.*c.*', ('Vlottende Activa', 'Vorderingen op groepsmaatschappijen')),
        (r'overige.*vordering|overlopende.*activa|nog.*te.*ontvangen', ('Vlottende Activa', 'Overige Vorderingen')),

        # === BALANS - VASTE ACTIVA ===
        (r'inventaris|bedrijfsmiddel|materieel|machine|gereedschap|computer|voertuig|auto|bestelauto|bus',
         ('Vaste Activa', 'Vaste Activa')),
        (r'afschr.*inventaris|afschr.*machine|afschr.*voertuig|afschr.*computer',
         ('Vaste Activa', 'Vaste Activa')),
        (r'verbouwing|inrichting|huurders.*investering', ('Vaste Activa', 'Vaste Activa')),

        # === BALANS - IMMATERIELE VASTE ACTIVA ===
        (r'goodwill', ('Immateriele Vaste Activa', 'Goodwill')),
        (r'immaterie|software.*licentie', ('Immateriele Vaste Activa', 'Goodwill')),

        # === BALANS - FINANCIELE VASTE ACTIVA ===
        (r'deelneming|participatie|aandelen.*dochter', ('Financiele Vaste Activa', 'Deelneming')),
        (r'waarborg|deposito|borg', ('Financiele Vaste Activa', 'Overige FVA')),

        # === BALANS - EIGEN VERMOGEN ===
        (r'aandelenkapitaal|aandelen.*kapitaal|geplaatst.*kapitaal|maatschappelijk.*kapitaal',
         ('Eigen vermogen', 'Geplaatst Kapitaal')),
        (r'agio|emissieopslag', ('Eigen vermogen', 'Agio Reserve')),
        (r'algemene.*reserve', ('Eigen vermogen', 'Algemene Reserve')),
        (r'overige.*reserve|wettelijke.*reserve|statutaire.*reserve', ('Eigen vermogen', 'Overige Reserve')),
        (r'resultaat.*lopend|resultaat.*boekjaar|winst.*boekjaar|resultaat\s+tm|resultaat\s+t\.m\.|resultaat\s+tot',
         ('Eigen vermogen', 'Resultaat Lopend jaar')),

        # === BALANS - VOORZIENINGEN ===
        (r'voorziening.*belasting|latente.*belasting', ('Voorzieningen', 'Voorzieningen latgente belastingen')),
        (r'voorziening(?!.*debiteuren)|reorganisatie.*voorziening', ('Voorzieningen', 'Overige voorzieningen')),

        # === BALANS - LANGLOPENDE SCHULDEN ===
        (r'hypotheek|lening.*bank|schuld.*bank', ('Langlopende schulden', 'Schulden Bank')),
        (r'achtergesteld|subordinated', ('Langlopende schulden', 'Schulden Achtergesteld')),
        (r'langlopend.*schuld|financiering', ('Langlopende schulden', 'Overige langlopende schulden')),

        # === BALANS - KORTLOPENDE SCHULDEN ===
        (r'crediteuren|crediteur|te betalen.*leverancier|schuld.*leverancier',
         ('Kortlopende schulden', 'Schulden aan leveranciers')),
        (r'btw|omzetbelasting|loonbelasting|loonheffing|vennootschapsbelasting|vpb|sociale.*premie|pensioenpremie',
         ('Kortlopende schulden', 'Belastingen en premies sociale verzekering')),
        (r'schuld.*groep|intercompany.*schuld', ('Kortlopende schulden', 'Schulden aan groepsmaatschappijen')),
        (r'vakantiegeld|vakantiedagen|vakantie.*verplichting|reservering.*vakantie|netto.*loon|overlopende.*passiva',
         ('Kortlopende schulden', 'Overige schulden en overlopende passiva')),
        (r'payroll|payrolling', ('Kortlopende schulden', 'Overige schulden en overlopende passiva')),
        (r'betaald.*pensioen|vrijwillig.*pensioen|pensioen.*verplichting|pensioenverplichting',
         ('Kortlopende schulden', 'Overige schulden en overlopende passiva')),
        (r'handelskrediet', ('Kortlopende schulden', 'Schulden handelskredieten')),

        # === ADDITIONAL BALANS PATTERNS (based on unmatched items analysis) ===
        # Meetinstrumenten, werktuigen
        (r'meet.*werktuig|werktuig|meetinstrument|meetapparatuur',
         ('Vaste Activa', 'Vaste Activa')),

        # Financial leases, vehicle financing
        (r'fin\.?\s|financ\.|financ\s|ford|transit|peugeot|lease|leasing',
         ('Kortlopende schulden', 'Overige schulden en overlopende passiva')),

        # Prepaid subscriptions, vooruitbetaald
        (r'vooruit|vooruitbetaald|vooruitgef|prepaid|abonnement',
         ('Vlottende Activa', 'Overige Vorderingen')),

        # Bank accounts with specific names (Rabobank, etc)
        (r'rabo|rabobank|abn|amro|ing\s|triodos',
         ('Vlottende Activa', 'Liquide Middelen')),

        # Nog te gebruiken (placeholder accounts)
        (r'nog te gebruiken|nog niet toegewezen|ongebruikt',
         ('Vlottende Activa', 'Overige Vorderingen')),

        # Generic resultaat fallback for WV (after Balans patterns above)
        (r'^resultaat', ('Financiele Baten en Lasten', 'Resultaat Deelneming (financiele baten en lasten)')),
    ]

    # Keyword-based fallback mappings (when no other match found)
    KEYWORD_FALLBACKS = [
        # === WV (Winst & Verlies) Keywords ===
        (['loon', 'salaris', 'personeel', 'medewerker'], ('Personeelkosten', 'Lonen en salarissen')),
        (['premie', 'sociale', 'verzekering', 'afdracht'], ('Personeelkosten', 'Sociale lasten')),
        (['pensioen', 'aow'], ('Personeelkosten', 'Pensioenlasten')),
        (['reis', 'vervoer', 'km', 'vergoeding'], ('Personeelkosten', 'Personeelskosten Overig')),
        (['uren', 'arbeid', 'werk'], ('Personeelkosten', 'Indirecte arbeidkosten')),
        (['afschrijving', 'afschr', 'deprec'], ('Afschrijving', 'Afschrijving')),
        (['omzet', 'verkoop', 'opbrengst'], ('Omzet', 'Omzet')),
        (['materiaal', 'inkoop', 'grondstof'], ('Directe kosten', 'Materiaal')),
        (['onderaannem', 'inleen', 'extern'], ('Directe kosten', 'Onderaanneming')),
        (['huur', 'huisvest', 'pand', 'gebouw'], ('Overige bedrijfskosten', 'Huisvestingkosten')),
        (['auto', 'voertuig', 'transport', 'brandstof'], ('Overige bedrijfskosten', 'Autokosten en overige Transportkosten')),
        (['kantoor', 'drukwerk', 'papier'], ('Overige bedrijfskosten', 'Kantoorkosten')),
        (['telefoon', 'internet', 'data', 'communicatie'], ('Overige bedrijfskosten', 'ICT en Communicatiekosten')),
        (['advies', 'accountant', 'juridisch'], ('Overige bedrijfskosten', 'Advies- en Accountantskosten')),
        (['marketing', 'reclame', 'promotie'], ('Overige bedrijfskosten', 'Marketing- en verkoopkosten')),
        (['rente', 'interest', 'financ'], ('Financiele Baten en Lasten', 'Financiele Baten en Lasten')),
        (['belasting', 'vpb', 'btw'], ('Belasting', 'Belasting')),

        # === BALANS Keywords ===
        (['voorraad', 'magazijn', 'goederen', 'materialen'], ('Vlottende Activa', 'Voorraden')),
        (['debiteur', 'vordering', 'te ontvangen'], ('Vlottende Activa', 'Debiteuren')),
        (['bank', 'kas', 'liquide', 'rekening', 'geld'], ('Vlottende Activa', 'Liquide Middelen')),
        (['crediteur', 'leverancier', 'te betalen'], ('Kortlopende schulden', 'Schulden aan leveranciers')),
        (['kapitaal', 'aandelen', 'reserve'], ('Eigen vermogen', 'Algemene Reserve')),
        (['hypotheek', 'lening'], ('Langlopende schulden', 'Schulden Bank')),
        (['deelneming', 'participatie'], ('Financiele Vaste Activa', 'Deelneming')),
        (['inventaris', 'machine', 'installatie'], ('Vaste Activa', 'Vaste Activa')),
        (['schuld', 'passiva', 'verplichting'], ('Kortlopende schulden', 'Overige schulden en overlopende passiva')),
        (['activa', 'bezit', 'vermogen'], ('Vlottende Activa', 'Overige Vorderingen')),
    ]

    # Default catch-all categories for completely unmatched items
    # Tries multiple options in order (first match wins)
    CATCH_ALL_CATEGORIES = [
        # WV (Winst & Verlies) catch-all
        ('Overige bedrijfskosten', 'Algemene kosten'),
        # Balans catch-all options
        ('Kortlopende schulden', 'Overige schulden en overlopende passiva'),
        ('Vlottende Activa', 'Overige Vorderingen'),
    ]

    def __init__(self, index: MappingIndex, use_catch_all: bool = False):
        self.index = index
        self.match_stats = Counter()
        self.use_catch_all = use_catch_all

        # Build niveau lookup from index
        self.niveau_to_codes: Dict[Tuple[str, str], List[Tuple]] = defaultdict(list)
        self._build_niveau_lookup()

    def _build_niveau_lookup(self):
        """Build lookup from (Niveau1, Niveau2) to available codes."""
        for key, combo in self.index._resolved.items():
            if combo and len(combo) >= 3:
                code, n1, n2 = combo[0], combo[1], combo[2]
                # Normalize for lookup
                n1_norm = n1.lower().strip() if n1 else ''
                n2_norm = n2.lower().strip() if n2 else ''
                self.niveau_to_codes[(n1_norm, n2_norm)].append(combo)

        # Sort each list by numeric code (lowest first)
        for key in self.niveau_to_codes:
            self.niveau_to_codes[key].sort(key=lambda c: self._numeric_code(c[0]))

    def _numeric_code(self, code: str) -> int:
        """Convert code to numeric for comparison."""
        try:
            return int(str(code).replace('.0', ''))
        except (ValueError, TypeError):
            return float('inf')

    def _find_by_niveau(self, niveau1: str, niveau2: Optional[str]) -> Optional[Tuple]:
        """Find best combination matching the given niveaus."""
        n1_norm = niveau1.lower().strip() if niveau1 else ''

        if niveau2:
            n2_norm = niveau2.lower().strip()
            combos = self.niveau_to_codes.get((n1_norm, n2_norm), [])
            if combos:
                return combos[0]  # Already sorted by lowest code

        # If no niveau2 or not found, search all with matching niveau1
        for (n1, n2), combos in self.niveau_to_codes.items():
            if n1 == n1_norm:
                if combos:
                    return combos[0]

        return None

    def match(self, normalized_key: str) -> Tuple[Optional[Tuple], str]:
        """
        Match a normalized rubriek key to a mapping combination.

        Returns:
            Tuple of (combination, match_method) or (None, 'unmatched')
        """
        # Step 1: Exact match
        result = self.index.get_exact(normalized_key)
        if result:
            self.match_stats['exact'] += 1
            return result, 'exact'

        # Step 2: Fixed anchors
        for pattern, fixed_combo in self.FIXED_ANCHORS:
            if re.search(pattern, normalized_key, re.IGNORECASE):
                # Verify the combination exists or is compatible
                self.match_stats['anchor_fixed'] += 1
                return fixed_combo, 'anchor_fixed'

        # Step 3: Niveau anchors
        for pattern, (niveau1, niveau2) in self.NIVEAU_ANCHORS:
            if re.search(pattern, normalized_key, re.IGNORECASE):
                combo = self._find_by_niveau(niveau1, niveau2)
                if combo:
                    self.match_stats['anchor_niveau'] += 1
                    return combo, 'anchor_niveau'

        # Step 4: Fuzzy match on candidates (threshold lowered to 85%)
        candidates = self.index.get_candidates(normalized_key)

        best_score = 0
        best_result = None

        for candidate_key in candidates:
            score = token_set_ratio(normalized_key, candidate_key)
            if score >= 85 and score > best_score:  # Lowered from 90% to 85%
                result = self.index.get_exact(candidate_key)
                if result:
                    best_score = score
                    best_result = result

        if best_result:
            self.match_stats['fuzzy'] += 1
            return best_result, 'fuzzy'

        # Step 5: Prefix/subterm fallback
        for candidate_key in candidates:
            if len(normalized_key) >= 4 and len(candidate_key) >= 4:
                if prefix_match(normalized_key, candidate_key) or subterm_match(normalized_key, candidate_key):
                    result = self.index.get_exact(candidate_key)
                    if result:
                        self.match_stats['prefix'] += 1
                        return result, 'prefix'

        # Step 6: Keyword-based fallback (best guess based on common keywords)
        tokens = set(normalized_key.lower().split())
        for keywords, (niveau1, niveau2) in self.KEYWORD_FALLBACKS:
            if any(kw in token or token.startswith(kw) for kw in keywords for token in tokens):
                combo = self._find_by_niveau(niveau1, niveau2)
                if combo:
                    self.match_stats['keyword_fallback'] += 1
                    return combo, 'keyword_fallback'

        # Step 7: Catch-all fallback (if enabled)
        # Tries multiple catch-all categories until one matches
        if self.use_catch_all:
            # First try preferred catch-all categories
            for niveau1, niveau2 in self.CATCH_ALL_CATEGORIES:
                combo = self._find_by_niveau(niveau1, niveau2)
                if combo:
                    self.match_stats['catch_all'] += 1
                    return combo, 'catch_all'

            # If no preferred category found, use any available combo from index
            # Prefer combos with common Niveau1 categories
            fallback_niveau1_order = [
                'Vlottende Activa', 'Kortlopende schulden', 'Vaste Activa',
                'Overige bedrijfskosten', 'Eigen vermogen'
            ]
            for n1 in fallback_niveau1_order:
                for (n1_key, n2_key), combos in self.niveau_to_codes.items():
                    if n1_key == n1.lower() and combos:
                        self.match_stats['catch_all'] += 1
                        return combos[0], 'catch_all'

            # Last resort: pick first available combo
            if self.niveau_to_codes:
                for combos in self.niveau_to_codes.values():
                    if combos:
                        self.match_stats['catch_all'] += 1
                        return combos[0], 'catch_all'

        self.match_stats['unmatched'] += 1
        return None, 'unmatched'
