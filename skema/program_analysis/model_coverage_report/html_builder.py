from pathlib import Path

from bs4 import BeautifulSoup

HTML_BASE_PATH = Path(__file__).resolve().parent / "base.html"
HTML_BASE = HTML_BASE_PATH.read_text()


class HTML_Instance:
    def __init__(self):
        self.soup = BeautifulSoup(HTML_BASE)

    def add_table_header_field(self, table_header_tag, field_name: str):
        """Adds a new field to a table header"""
        field_tag = self.soup.new_tag("th")
        field_tag.string = field_name

        table_header_tag.append(field_tag)

    def add_table_data_field(self, table_row_tag, field_data: str, anchored=False, anchor_text=None):
        """Adds a new field to a table row"""
        field_tag = self.soup.new_tag("td")

        if anchored:
            anchor_tag = self.soup.new_tag("a")
            anchor_tag["href"] = field_data
            anchor_tag.string = anchor_text
            field_tag.append(anchor_tag)
        else:
            field_tag.string = field_data

        table_row_tag.append(field_tag)

    def add_model(self, model_name: str):
        """Adds a new model to the HTML source"""
        # Create the new model HTML structure
        new_model_container = self.soup.new_tag(
            "div", id=model_name, class_="model-container"
        )
        new_model_heading = self.soup.new_tag("h2")
        new_model_heading.string = model_name
        # Create basic model table
        new_model_table_container_basic = self.soup.new_tag(
            "div", id=f"{model_name}-basic", class_="table-container"
        )

        new_model_table_basic = self.soup.new_tag("table", _class="searchable sortable")
        new_model_thead = self.soup.new_tag("thead")

        new_model_table_header_basic = self.soup.new_tag("tr")
        self.add_table_header_field(new_model_table_header_basic, "File Name")
        self.add_table_header_field(new_model_table_header_basic, "Num Lines")
        self.add_table_header_field(new_model_table_header_basic, "Can Ingest")
        self.add_table_header_field(new_model_table_header_basic, "Tree-Sitter Parse Tree")
        self.add_table_header_field(new_model_table_header_basic, "CAST")
        self.add_table_header_field(new_model_table_header_basic, "Gromet")
        self.add_table_header_field(new_model_table_header_basic, "Gromet Errors")
        self.add_table_header_field(new_model_table_header_basic, "Gromet Report")
        self.add_table_header_field(new_model_table_header_basic, "Preprocessed Gromet")
        
        # Append the elements to each other
        new_model_container.extend([new_model_heading, new_model_table_container_basic])
        new_model_thead.append(new_model_table_basic)
        new_model_table_container_basic.append(new_model_thead)
        new_model_table_basic.append(new_model_table_header_basic)


        # Append to outer body
        self.soup.body.append(new_model_container)

    def add_model_header_data(
        self,
        model: str,
        supported_lines: int,
        total_lines: int,
        full_ingest: bool,
        full_ingest_path: str,
    ):
        """Adds header data such as ingestion percentage to a mode"""
        # Calculate ingestion percentage based on supported and total lines
        try:
            ingestion_percentage = (supported_lines / total_lines) * 100
        except ZeroDivisionError as e:
            ingestion_percentage = 0

        model_table = self.soup.select_one(f"#{model}-basic table")

        model_header = self.soup.new_tag("div", class_="model_header")
        model_header_supported_lines = self.soup.new_tag("div", class_="model_header")
        model_header_supported_lines.string = f"Total Supported Lines: {supported_lines}/{total_lines} ({ingestion_percentage}%)"
        model_header_full_ingest = self.soup.new_tag("div")
        model_header_full_ingest.string = f"Can ingest system into single GrometFNModuleCollection: {str(full_ingest)}"
        model_header_full_system_a = self.soup.new_tag("a")
        model_header_full_system_a["href"] = full_ingest_path
        model_header_full_system_a.string = "Open Full System Gromet"

        model_header.extend(
            [
                model_header_supported_lines,
                model_header_full_ingest,
                model_header_full_system_a,
            ]
        )
        model_table.append(model_header)

    def add_file_basic(
        self,
        model: str,
        file_name: str,
        num_lines: int,
        can_ingest: bool,
        parse_tree_path: Path,
        cast_path: Path,
        gromet_path: Path,
        gromet_errors: int = 0,
        gromet_report_path: Path = Path(""),
        preprocessed_gromet_path: Path = Path("")
    ):
        """Add a file entry to a model table"""
        model_table = self.soup.select_one(f"#{model}-basic table")
        new_row = self.soup.new_tag("tr")

        # Add row data fields
        self.add_table_data_field(new_row, file_name)
        self.add_table_data_field(new_row, str(num_lines))
        self.add_table_data_field(new_row, "✓" if can_ingest else "✗")
        self.add_table_data_field(new_row, str(parse_tree_path), anchored=True, anchor_text="Open Parse Tree")
        self.add_table_data_field(new_row, str(cast_path), anchored=True, anchor_text="Open CAST")
        self.add_table_data_field(new_row, str(gromet_path), anchored=True, anchor_text="Open Gromet")
        self.add_table_data_field(new_row, str(gromet_errors))
        self.add_table_data_field(new_row, str(gromet_report_path), anchored=True, anchor_text="Open Gromet Report")
        self.add_table_data_field(new_row, str(preprocessed_gromet_path), anchored=True, anchor_text="Open Preprocessed Gromet")
        
        model_table.append(new_row)

    def write_html(self):
        """Output html to a file."""
        output_path = Path(__file__).resolve().parent / "output.html"
        output_path.write_text(self.soup.prettify())
