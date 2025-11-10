from pathlib import Path
import pandas as pd
import re
import string
import time
import json
from difflib import SequenceMatcher


class ScientificTextToMarkdownConverter:
    """
    Converts OCR'd scientific paper text into a structured Markdown file.
    Integrates with Excel file to prepend abstract and identifies main sections.
    """

    SECTION_PATTERNS = {
        "Introduction": [
            r"introduction",
            r"i\s*ntroduction",
            r"intro\s*duction",
            r"\d+\.?\s*introduction",
        ],
        "Materials and Methods": [
            r"materials\s+and\s+methods",
            r"material\s+and\s+methods",
            r"materials\s+&\s+methods",
            r"methods\s+and\s+materials",
            r"experimental\s+section",
            r"experimental\s+details",
            r"experimental\s+procedures",
            r"experimental\s+methods",
            r"experimental",
            r"methods",
            r"methodology",
            r"methodologies",
            r"\d+\.?\s*materials\s+and\s+methods",
            r"\d+\.?\s*methods",
            r"\d+\.?\s*experimental",
        ],
        "Results and Discussion": [
            # Prefer joint/full forms first
            r"results\s+and\s+discussion",
            r"results\s+&\s+discussion",
            r"result\s+and\s+discussion",
            r"\d+\.?\s*results\s+and\s+discussion",
            # allow single-part words too (we'll post-process)
            r"results",
            r"result",
            r"\d+\.?\s*results?",
            r"discussion",
            r"discussions",
            r"\d+\.?\s*discussions?",
        ],
        "Results": [
            r"\bresults?\b",
            r"\bresult\b",
            r"\d+\.?\s*results?",
        ],
        "Discussion": [
            r"\bdiscussion\b",
            r"\bdiscussions\b",
            r"\d+\.?\s*discussions?",
        ],
        "Conclusion": [
            r"conclusion",
            r"conclusions",
            r"concluding\s+remarks",
            r"summary\s+and\s+conclusion",
            r"\d+\.?\s*conclusions?",
        ],
        "Acknowledgements": [
            r"acknowledgements?",
            r"acknowledgments?",
            r"funding",
            r"financial\s+support",
            r"\d+\.?\s*acknowledgements?",
        ],
        "References": [
            r"references",
            r"bibliography",
            r"literature\s+cited",
            r"works\s+cited",
            r"\d+\.?\s*references",
        ],
    }

    def __init__(self, ocr_filepath: Path, excel_filepath: Path, log_file: Path = None):
        self.ocr_filepath = Path(ocr_filepath)
        self.excel_filepath = Path(excel_filepath)
        self.log_file = log_file
        self.match_found = False
        self.processed_text = None
        self.abstract_text = None
        self.log_entries = []
        self.section_report = {
            "filename": self.ocr_filepath.stem,
            "title_found": False,
            "sections_found": [],
            "abstract_source": None,
            "processing_notes": []
        }

        with open(self.ocr_filepath, 'r', encoding='utf-8') as f:
            ocr_text = f.read()

        self._get_abstract_from_excel()
        ocr_text = self._remove_title_metadata(ocr_text)
        ocr_text = self._clean_text_before_introduction(ocr_text)  # Add this line
        ocr_text = self._identify_and_format_sections(ocr_text)

        if self.abstract_text:
            ocr_text = f"# Abstract\n\n{self.abstract_text}\n\n{ocr_text}"
            self.section_report["sections_found"].insert(0, {
                "section": "Abstract",
                "line": 0,
                "method": "prepended_from_excel"
            })

        self.processed_text = ocr_text

        if self.log_file and self.log_entries:
            self._write_log()

    def _log(self, message: str):
        self.log_entries.append(message)
        print(message)

    def _write_log(self):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(self.log_entries) + '\n\n')
        except Exception as e:
            print(f"    Warning: Could not write to log file: {e}")

    def _create_comparison_string(self, text: str) -> str:
        text = text.lower()
        allowed = string.ascii_lowercase + string.digits + ' '
        text = ''.join(c if c in allowed else ' ' for c in text)
        text = ' '.join(text.split())
        return text

    def _find_best_match(self, df: pd.DataFrame, paper_name: str) -> int:
        """Find the best matching row for the paper name in the DataFrame."""
        normalized_paper_name = self._create_comparison_string(paper_name)

        self._log(f"    Searching for: '{paper_name}'")
        self._log(f"    Normalized to: '{normalized_paper_name}'")

        # First pass: exact match (use df.items to preserve index labels)
        for idx, title in df['Title'].astype(str).items():
            # robust NaN check
            if pd.isna(title) or str(title).strip().lower() in ('nan', ''):
                continue
            normalized_title = self._create_comparison_string(title)
            if normalized_paper_name == normalized_title:
                self._log(f"    ✓ Exact match found!")
                return idx

        self._log(f"    No exact match, trying fuzzy matching...")

        # Second pass: combined fuzzy matching
        paper_words = set(normalized_paper_name.split())
        best_match_idx = None
        best_similarity = 0.0
        best_title = ""

        for idx, title in df['Title'].astype(str).items():
            if pd.isna(title) or str(title).strip().lower() in ('nan', ''):
                continue
            normalized_title = self._create_comparison_string(title)
            title_words = set(normalized_title.split())

            # Token Jaccard
            union = len(paper_words | title_words) if (paper_words or title_words) else 1
            intersection = len(paper_words & title_words)
            jaccard = intersection / union if union > 0 else 0

            # Containment (how much paper_name's tokens are contained in title)
            containment = intersection / len(paper_words) if len(paper_words) > 0 else 0

            # Sequence similarity (for order/character-level similarity)
            seq_ratio = SequenceMatcher(None, normalized_paper_name, normalized_title).ratio()

            # Take the strongest signal
            final_similarity = max(jaccard, containment, seq_ratio)

            if final_similarity > best_similarity:
                best_similarity = final_similarity
                best_match_idx = idx
                best_title = title

        # Accept if similarity is high enough. Slightly relaxed threshold to avoid misses.
        if best_match_idx is not None and best_similarity >= 0.80:
            self._log(f"    ✓ Fuzzy match found (similarity: {best_similarity:.2%})")
            self._log(f"    Matched to: '{best_title}'")
            return best_match_idx

        if best_match_idx is not None:
            self._log(f"    Best match was '{best_title}' but similarity too low: {best_similarity:.2%}")

        return -1

    def _get_abstract_from_excel(self):
        max_retries = 3
        df = None
        for attempt in range(max_retries):
            try:
                df = pd.read_excel(self.excel_filepath)
                break
            except PermissionError:
                if attempt == max_retries - 1:
                    self._log(f"    ✗ Error: Cannot access Excel file.")
                    return
                time.sleep(1)

        if df is None or 'Title' not in df.columns or 'Abstract' not in df.columns:
            self._log(f"    ✗ Excel missing required columns.")
            return

        paper_name = re.sub(r'_attempt_?\d+$', '', self.ocr_filepath.stem)
        row_idx = self._find_best_match(df, paper_name)
        if row_idx == -1:
            self._log(f"    ✗ No match found in Excel.")
            self.section_report["abstract_source"] = "none"
            return

        self.match_found = True
        self._log(f"    ✓ Found match: '{df.loc[row_idx, 'Title']}'")

        abstract_text = df.loc[row_idx, 'Abstract']
        if pd.isna(abstract_text) or not str(abstract_text).strip():
            self.abstract_text = None
            self.section_report["abstract_source"] = "excel_empty"
            self._log("    Note: Abstract is empty for this paper.")
        else:
            self.abstract_text = str(abstract_text).strip()
            self.section_report["abstract_source"] = "excel"
            self._log(f"    ✓ Retrieved abstract from Excel ({len(self.abstract_text)} chars)")

        # Mark processed flag if present / create if not
        if 'Processed' not in df.columns:
            df['Processed'] = False
        else:
            if df['Processed'].dtype != 'bool':
                df['Processed'] = df['Processed'].fillna(False).astype('object')

        df.at[row_idx, 'Processed'] = True

        for attempt in range(max_retries):
            try:
                df.to_excel(self.excel_filepath, index=False)
                self._log(f"    ✓ Marked as processed in Excel")
                break
            except PermissionError:
                if attempt == max_retries - 1:
                    self._log(f"    ✗ Warning: Could not save Excel file.")
                else:
                    time.sleep(1)

    def _remove_title_metadata(self, text: str) -> str:
        self._log("  Cleaning title/metadata... (keeping all content)")
        return text

    def _is_reference_line(self, line: str) -> bool:
        line = line.strip()
        if not line or len(line) < 10:
            return False
        patterns = [
            r'^\s*\d+[\.\)\-]\s+',
            r'^\s*\[\d+\]',
            r'\b(19|20)\d{2}\b',
            r'\bet\s+al\.?',
            r'\bdoi\s*:',
            r'https?://',
        ]
        return any(re.search(p, line, re.IGNORECASE) for p in patterns)

    def _find_references_section(self, lines: list):
        self._log("    Searching for References section...")
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip().lower()
            if len(line) > 150 or len(line) < 3:
                continue
            for pattern in self.SECTION_PATTERNS["References"]:
                if re.fullmatch(pattern, line, re.IGNORECASE):
                    self._log(f"    ✓ Found explicit References at line {i}")
                    return i, "explicit"

        consecutive = 0
        start = -1
        for i in range(len(lines) - 1, max(len(lines) - 100, -1), -1):
            if self._is_reference_line(lines[i]):
                consecutive += 1
                start = i
            elif lines[i].strip():
                if consecutive >= 5:
                    break
                consecutive = 0
        if consecutive >= 5:
            self._log(f"    ✓ Found implicit References at line {start}")
            return start, "implicit"
        return -1, None

    def _find_explicit_section(self, lines, section_name, start_line=0, end_line=None):
        """
        Find explicit heading using patterns for a given section name.
        Returns (line_index, raw_heading_text) or (-1, None) if not found.
        """
        if end_line is None:
            end_line = len(lines)
        target_patterns = self.SECTION_PATTERNS.get(section_name, [])
        exact_matches = []

        # Pass 1: Exact phrase matches
        for i in range(start_line, end_line):
            line = lines[i].strip().lower()
            if not line or len(line) > 150:
                continue
            for phrase in [re.sub(r'\\d\+.*', '', p).replace('\\s+', ' ') for p in target_patterns]:
                clean_phrase = re.sub(r'[^a-z ]', '', phrase).strip()
                if clean_phrase and line == clean_phrase:
                    exact_matches.append(i)

        if len(exact_matches) == 1:
            i = exact_matches[0]
            self._log(f"    ✓ Exact {section_name} heading found at line {i}")
            return i, lines[i]

        # Pass 2: Regex and fuzzy fallback
        best_line, best_score, best_text = -1, 0, ""
        for i in range(start_line, end_line):
            line = lines[i].strip()
            if not line or len(line) > 150:
                continue
            l = line.lower()
            for pattern in target_patterns:
                # try anchored regex
                try:
                    if re.match(f'^{pattern}', l, re.IGNORECASE):
                        self._log(f"    ✓ Regex {section_name} match at line {i}: '{line}'")
                        return i, line
                except re.error:
                    # skip invalid pattern
                    continue
                # Fuzzy match (fallback)
                ratio = SequenceMatcher(None, l, pattern.replace('\\s+', ' ')).ratio()
                if ratio > best_score:
                    best_score, best_line, best_text = ratio, i, line

        if best_score > 0.8:
            self._log(f"    ✓ Fuzzy {section_name} heading at line {best_line} (score {best_score:.2f})")
            return best_line, best_text

        return -1, None

    def _identify_and_format_sections(self, text: str) -> str:
        self._log("  Identifying sections...")
        lines = text.split('\n')
        sections = []
        search_start = 0

        # 1) Find references first
        ref_line, ref_method = self._find_references_section(lines)
        search_end = ref_line if ref_line != -1 else len(lines)

        # 2) Decide whether to search for joint Results-and-Discussion first,
        #    and only search for single Results / Discussion if joint not found.
        # Check joint explicitly first
        joint_line, joint_text = self._find_explicit_section(lines, "Results and Discussion", search_start, search_end)

        # Build the section scanning order dynamically to avoid duplicate/conflicting detections
        if joint_line != -1:
            # If joint exists, skip individual Results/Discussion searches
            section_order = [
                "Introduction",
                "Materials and Methods",
                "Results and Discussion",
                "Conclusion",
                "Acknowledgements",
            ]
        else:
            # Joint not found: allow searching for separate Results and Discussion (to possibly merge later)
            section_order = [
                "Introduction",
                "Materials and Methods",
                "Results",           # separate search
                "Discussion",        # separate search
                "Conclusion",
                "Acknowledgements",
            ]

        # collect explicit/fuzzy detections
        for section_name in section_order:
            try:
                line_num, heading_text = self._find_explicit_section(lines, section_name, search_start, search_end)
            except Exception as e:
                # Safety: do not let exceptions in detection interrupt the whole processing
                self._log(f"    Warning: Exception while searching for '{section_name}': {e}")
                line_num, heading_text = -1, None

            if line_num != -1:
                # avoid duplicate entries for same line
                if not any(s['line'] == line_num and s['name'] == section_name for s in sections):
                    sections.append({
                        "name": section_name,
                        "line": line_num,
                        "method": "explicit",
                        "original_text": heading_text
                    })
                    self.section_report["sections_found"].append({
                        "section": section_name,
                        "line": line_num,
                        "method": "explicit",
                        "text": heading_text
                    })

        # If Introduction missing, try to infer an implicit introduction
        sections.sort(key=lambda x: x['line'])
        if not any(s['name'] == 'Introduction' for s in sections):
            next_line = sections[0]['line'] if sections else search_end
            for i in range(search_start, next_line):
                if lines[i].strip() and len(lines[i].strip()) > 30:
                    sections.insert(0, {"name": "Introduction", "line": i, "method": "implicit"})
                    self._log(f"    ✓ Implicit Introduction at line {i}")
                    break

        # Add references if found
        if ref_line != -1:
            sections.append({"name": "References", "line": ref_line, "method": ref_method})
            self.section_report["sections_found"].append({
                "section": "References",
                "line": ref_line,
                "method": ref_method
            })

        # Sort detected sections before post-processing
        sections.sort(key=lambda x: x['line'])

        # --- POST-PROCESSING RULES ---
        # If joint_results was found earlier ensure we only keep the joint and remove proximate Results/Discussion
        try:
            if joint_line != -1:
                # ensure the joint entry is present (if not already added)
                if not any(s['name'] == 'Results and Discussion' for s in sections):
                    sections.append({"name": "Results and Discussion", "line": joint_line, "method": "explicit", "original_text": joint_text})
                    self.section_report["sections_found"].append({
                        "section": "Results and Discussion",
                        "line": joint_line,
                        "method": "explicit",
                        "text": joint_text
                    })
                # remove proximate single Results or Discussion headings (within small distance)
                cleaned = []
                for s in sections:
                    if s['name'] in ('Results', 'Discussion'):
                        if abs(s['line'] - joint_line) <= 3:
                            self._log(f"    ✓ Removing proximate duplicate heading '{s['name']}' near joint Results and Discussion at line {joint_line}")
                            continue
                    cleaned.append(s)
                sections = cleaned
            else:
                # No joint heading — if both Results AND Discussion present and are close, merge them
                res_idx = next((i for i, s in enumerate(sections) if s['name'] == 'Results'), None)
                disc_idx = next((i for i, s in enumerate(sections) if s['name'] == 'Discussion'), None)
                if res_idx is not None and disc_idx is not None:
                    rline = sections[res_idx]['line']
                    dline = sections[disc_idx]['line']
                    # merge threshold: if they are relatively close (few lines apart), merge them
                    if abs(rline - dline) <= 8:
                        merged_line = min(rline, dline)
                        # remove the original two and add merged
                        sections = [s for i, s in enumerate(sections) if i not in (res_idx, disc_idx)]
                        sections.append({"name": "Results and Discussion", "line": merged_line, "method": "merged"})
                        self._log(f"    ✓ Merged Results (line {rline}) and Discussion (line {dline}) into Results and Discussion")
        except Exception as e:
            # safety: never interrupt the pipeline because of post-processing
            self._log(f"    Warning: Exception during post-processing of sections: {e}")

        # Final sort
        sections = sorted(sections, key=lambda x: x['line'])

        if not sections:
            self._log("    ✗ No sections found.")
            return text

        # Build result text using the sections we finalized
        result = []
        pos = 0
        for section in sections:
            # append original lines up to section start
            result.extend(lines[pos:section['line']])
            # insert canonical heading line
            result.append(f"\n# {section['name']}\n")
            # when explicit, skip heading line to avoid repeating it
            pos = section['line'] + (1 if section.get('method', '') == 'explicit' else 0)
        result.extend(lines[pos:])
        self._log(f"    ✓ Identified {len(sections)} sections total")
        return '\n'.join(result)

    def get_text(self) -> str:
        return self.processed_text

    def save_to_file(self, output_path: Path) -> bool:
        if not self.match_found:
            self._log(f"    ✗ Skipping file creation (no Excel match)")
            return False
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.processed_text)
        self._log(f"    ✓ Saved to '{output_path.name}'")
        return True

    def save_section_report(self, report_path: Path):
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.section_report, f, indent=2, ensure_ascii=False)
            self._log(f"    ✓ Section report saved to '{report_path.name}'")
        except Exception as e:
            self._log(f"    ✗ Error saving section report: {e}")

    def _clean_text_before_introduction(self, text: str) -> str:
        """
        If Introduction section is found explicitly, removes all text before it.
        Otherwise returns the text as is.
        """
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip().lower()
            if not line or len(line) > 150:
                continue
            for pattern in self.SECTION_PATTERNS["Introduction"]:
                try:
                    if re.match(f'^{pattern}', line, re.IGNORECASE):
                        self._log(f"    ✓ Found Introduction at line {i}, cleaning preceding text...")
                        return '\n'.join(lines[i:])
                except re.error:
                    continue
        return text
