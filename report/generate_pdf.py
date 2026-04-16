"""Generate the PoC report as PDF."""
from fpdf import FPDF


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "MoSPI - RAG with Small Language Models - PoC Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(0, 51, 102)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 51, 102)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(4)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold_body(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(6, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(240, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            max_h = 7
            for i, cell in enumerate(row):
                self.cell(col_widths[i], max_h, str(cell), border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(3)


def build():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 12, "RAG with Small Language Models", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "for MoSPI Documents", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, "Proof of Concept Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 6, "Prepared by: Satvik Bajpai  |  Computer Centre, MoSPI  |  April 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # 1. Summary
    pdf.section("1. Objective")
    pdf.body(
        "Test whether Small Language Models (under 8B parameters) can power MoSPI's "
        "document chatbots using RAG, running on a standard office desktop with no GPU - "
        "replacing the current H200 GPU-hosted solution."
    )
    pdf.bold_body("Result: Gemma 3 4B achieved 100% accuracy (16/16) on our eval set, running on a standard HP desktop (Intel 13th Gen, 16 GB RAM, no GPU).")

    # 2. Architecture
    pdf.section("2. Architecture")
    pdf.body("User Query -> Embedding (BGE-small-en-v1.5) -> Vector Search (ChromaDB, top-5 chunks) -> Prompt Assembly -> SLM via Ollama -> Answer with source citations")
    pdf.body("All components run locally on a single machine. No external APIs, no GPU, no cloud dependency.")

    # 3. Knowledge Base
    pdf.section("3. Knowledge Base (10 Documents)")
    pdf.table(
        ["#", "Document", "Domain", "Period"],
        [
            ["1", "CPI_Nov2025.pdf", "CPI", "November 2025"],
            ["2", "CPI_Dec2025.pdf", "CPI", "December 2025"],
            ["3", "CPI_Jan2026.pdf", "CPI", "January 2026"],
            ["4", "CPI_Feb2026.pdf", "CPI", "February 2026"],
            ["5", "IIP_Nov2025.pdf", "IIP", "November 2025"],
            ["6", "IIP_Dec2025.pdf", "IIP", "December 2025"],
            ["7", "IIP_Feb2026.pdf", "IIP", "February 2026"],
            ["8", "PLFS_JulSep2024.pdf", "PLFS", "Jul-Sep 2024"],
            ["9", "PLFS_OctDec2024.pdf", "PLFS", "Oct-Dec 2024"],
            ["10", "PLFS_OctDec2025.pdf", "PLFS", "Oct-Dec 2025"],
        ],
        col_widths=[10, 70, 30, 80],
    )
    pdf.body("All PDFs downloaded from mospi.gov.in press release section. After extraction and chunking: 204 text chunks indexed.")

    # 4. Eval Queries
    pdf.section("4. Evaluation Queries (16 Questions)")
    pdf.table(
        ["#", "Question", "Expected Answer"],
        [
            ["1", "CPI headline inflation, Feb 2026?", "3.21%"],
            ["2", "Food inflation (CFPI), Feb 2026?", "3.47%"],
            ["3", "Housing inflation, Feb 2026?", "2.12%"],
            ["4", "CPI inflation, Jan 2026?", "2.75%"],
            ["5", "CPI headline inflation, Dec 2025?", "1.33%"],
            ["6", "Food inflation, Nov 2025?", "-3.91%"],
            ["7", "IIP growth rate, Feb 2026?", "5.2%"],
            ["8", "Manufacturing growth, Feb 2026 IIP?", "6.0%"],
            ["9", "IIP growth rate, Dec 2025?", "7.8%"],
            ["10", "Manufacturing growth, Dec 2025?", "8.1%"],
            ["11", "IIP growth rate, Nov 2025?", "6.7%"],
            ["12", "Manufacturing growth, Nov 2025?", "8.0%"],
            ["13", "LFPR (15+), Oct-Dec 2025?", "55.8%"],
            ["14", "Female LFPR, Oct-Dec 2025?", "34.9%"],
            ["15", "WPR (15+), Oct-Dec 2025?", "53.1%"],
            ["16", "GDP growth Q3 FY26? (out-of-corpus)", "Should refuse"],
        ],
        col_widths=[10, 105, 75],
    )

    # 5. Results
    pdf.section("5. Model Comparison Results")
    pdf.table(
        ["Model", "Params", "Accuracy", "Retrieval", "Latency", "Refuses OOC?"],
        [
            ["Qwen 2.5", "0.5B", "11/16 (69%)", "16/16", "~27s", "No"],
            ["Phi-4-mini", "3.8B", "12/16 (75%)", "16/16", "~52s", "Yes"],
            ["Llama 3", "8B", "13/16 (81%)", "16/16", "~87s", "Yes"],
            ["Llama 3.2", "3B", "14/16 (88%)", "16/16", "~40s", "Yes"],
            ["Gemma 3", "4B", "16/16 (100%)", "16/16", "~62s", "Yes"],
        ],
        col_widths=[30, 20, 35, 25, 25, 30],
    )
    pdf.bold_body("Key findings:")
    pdf.bullet("Retrieval accuracy is 100% across all models - the right document is always found.")
    pdf.bullet("Gemma 3 4B (Google) achieves perfect accuracy including the out-of-corpus refusal test.")
    pdf.bullet("Larger models are not always better: Llama 3.2 3B (88%) outperforms Llama 3 8B (81%).")
    pdf.bullet("Smaller models (0.5B) hallucinate when the answer is not in the corpus - unacceptable for government use.")

    # 6. Infra comparison
    pdf.section("6. Infrastructure Comparison")
    pdf.table(
        ["", "Current (H200 GPU)", "SLM on Standard Server"],
        [
            ["Hardware", "NVIDIA H200 GPU", "4-core CPU, 16 GB RAM"],
            ["Model Size", "20B+ parameters", "3-4B parameters"],
            ["GPU Required", "Yes", "No"],
            ["Est. Monthly Cost", "Rs 1.5-2.5 lakh", "Rs 2,000-5,000"],
            ["Answer Quality", "Baseline", "100% on eval set (Gemma 3 4B)"],
            ["Latency", "1-2 seconds", "~60 seconds (CPU)"],
        ],
        col_widths=[45, 72, 73],
    )

    # 7. Hardware tested
    pdf.section("7. Hardware Used for Testing")
    pdf.body("HP Pro Tower 280 G9 E PCI Desktop PC")
    pdf.bullet("Processor: Intel 13th Gen (Core i5), 2100 MHz")
    pdf.bullet("RAM: 16 GB DDR4")
    pdf.bullet("GPU: None (integrated graphics only)")
    pdf.bullet("OS: Windows 11 Pro")
    pdf.bullet("SLM Runtime: Ollama (4-bit quantized models)")

    # 8. Recommendations
    pdf.section("8. Recommendations")
    pdf.bold_body("For production deployment:")
    pdf.bullet("Use Gemma 3 4B via Ollama on a standard NIC server (4 vCPU, 16 GB RAM).")
    pdf.bullet("Deploy as Docker Compose (FastAPI app + Ollama + ChromaDB) - no GPU infra needed.")
    pdf.bullet("Expand corpus to all MoSPI press releases, annual reports, and survey docs.")
    pdf.bullet("Add a web UI (Streamlit/Gradio) for non-technical users to evaluate.")
    pdf.bullet("Benchmark against the existing StatsDoc chatbot with a shared test set.")
    pdf.ln(4)
    pdf.bold_body("Repository: https://github.com/SatvikBajpai/mospi-rag-poc")

    pdf.output("report/SLM_RAG_PoC_Report.pdf")
    print("PDF saved to report/SLM_RAG_PoC_Report.pdf")


if __name__ == "__main__":
    build()
