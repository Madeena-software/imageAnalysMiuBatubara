"""
PDF Export Module for Image Analysis Tool
Generates PDF reports with analysis results, images, and statistics
"""

import io
import base64
from datetime import datetime

# Note: fpdf2 is used for PDF generation in PyScript/Pyodide environment
from fpdf import FPDF


class ImageAnalysisPDF(FPDF):
    """Custom PDF class for Image Analysis reports"""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        """PDF Header"""
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(21, 114, 232)  # Blue color matching the app
        self.cell(0, 10, "Image Analysis Report", border=0, ln=True, align="C")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(100, 100, 100)
        self.cell(
            0,
            5,
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            border=0,
            ln=True,
            align="C",
        )
        self.ln(5)
        self.set_draw_color(21, 114, 232)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)

    def footer(self):
        """PDF Footer"""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(
            0, 10, f"Page {self.page_no()}/{{nb}} - Image Analysis Tool", align="C"
        )

    def add_section_title(self, title):
        """Add a section title"""
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(104, 97, 206)  # Purple color
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def add_subsection_title(self, title):
        """Add a subsection title"""
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(87, 89, 98)
        self.cell(0, 8, title, ln=True)
        self.ln(1)

    def add_text(self, text):
        """Add normal text"""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(87, 89, 98)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def add_key_value(self, key, value):
        """Add a key-value pair"""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(87, 89, 98)
        self.cell(60, 6, f"{key}:", ln=False)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, str(value), ln=True)

    def add_image_from_base64(self, base64_string, width=180, caption=None):
        """Add an image from base64 string"""
        try:
            # Decode base64 to bytes
            img_data = base64.b64decode(base64_string)

            # Create a temporary file-like object
            img_buffer = io.BytesIO(img_data)

            # Calculate position to center the image
            x_pos = (210 - width) / 2  # A4 width is 210mm

            # Add image
            self.image(img_buffer, x=x_pos, w=width)

            # Add caption if provided
            if caption:
                self.ln(2)
                self.set_font("Helvetica", "I", 9)
                self.set_text_color(128, 128, 128)
                self.cell(0, 5, caption, ln=True, align="C")

            self.ln(5)
            return True
        except Exception as e:
            self.add_text(f"[Image could not be loaded: {str(e)}]")
            return False

    def add_table(self, headers, data, col_widths=None):
        """Add a table to the PDF"""
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Table header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(21, 114, 232)
        self.set_text_color(255, 255, 255)

        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, border=1, align="C", fill=True)
        self.ln()

        # Table data
        self.set_font("Helvetica", "", 9)
        self.set_text_color(87, 89, 98)
        fill = False

        for row in data:
            if fill:
                self.set_fill_color(249, 251, 253)
            else:
                self.set_fill_color(255, 255, 255)

            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), border=1, align="C", fill=True)
            self.ln()
            fill = not fill

        self.ln(5)


def generate_circle_detection_pdf(
    filename,
    result,
    params,
    grid_result=None,
    histogram_result=None,
    diagonal_result=None,
):
    """
    Generate PDF report for circle detection analysis.

    Parameters:
    -----------
    filename : str
        Original filename of the analyzed image
    result : dict
        Circle detection results
    params : dict
        Processing parameters used
    grid_result : dict, optional
        Grid detection results
    histogram_result : dict, optional
        Histogram analysis results
    diagonal_result : dict, optional
        Diagonal comparison results

    Returns:
    --------
    str : Base64 encoded PDF
    """
    try:
        pdf = ImageAnalysisPDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        # Title section
        pdf.add_section_title("Circle Detection Analysis")
        pdf.add_key_value("Source File", filename)
        pdf.add_key_value("Circles Detected", result.get("count", 0))
        pdf.ln(5)

        # Parameters section
        pdf.add_subsection_title("Processing Parameters")
        pdf.add_key_value("Threshold Value", params.get("threshold_value", "N/A"))
        pdf.add_key_value("Min Diameter", f"{params.get('min_diameter', 'N/A')} px")
        pdf.add_key_value("Max Diameter", f"{params.get('max_diameter', 'N/A')} px")
        pdf.add_key_value("Min Circularity", params.get("min_circularity", "N/A"))
        pdf.add_key_value("Min Solidity", params.get("min_solidity", "N/A"))
        pdf.add_key_value("Expected Count", params.get("expected_count", "N/A"))
        pdf.ln(5)

        # Detection image
        if "detection_image" in result:
            pdf.add_subsection_title("Detection Result")
            pdf.add_image_from_base64(
                result["detection_image"],
                width=160,
                caption="Detected circles with labels",
            )

        # Mask image
        if "mask_image" in result:
            pdf.add_subsection_title("Threshold Mask")
            pdf.add_image_from_base64(
                result["mask_image"],
                width=160,
                caption=f"Binary mask (threshold < {params.get('threshold_value', 'N/A')})",
            )

        # Circle statistics table
        if "circles" in result and len(result["circles"]) > 0:
            pdf.add_page()
            pdf.add_section_title("Circle Statistics")

            headers = [
                "#",
                "Grid Pos",
                "Center (X,Y)",
                "Diameter",
                "Mean Value",
                "Classification",
            ]
            data = []

            for i, circle in enumerate(result["circles"]):
                grid_pos = circle.get("grid_pos", "N/A")
                if isinstance(grid_pos, (list, tuple)):
                    grid_pos = f"[{grid_pos[0]}, {grid_pos[1]}]"

                center = circle.get("center", ["N/A", "N/A"])
                center_str = f"({center[0]}, {center[1]})"

                data.append(
                    [
                        str(i + 1),
                        str(grid_pos),
                        center_str,
                        f"{circle.get('diameter', 'N/A'):.1f}",
                        f"{circle.get('mean_value', 'N/A'):.1f}",
                        circle.get("classification", "N/A"),
                    ]
                )

            col_widths = [15, 25, 45, 30, 35, 40]
            pdf.add_table(headers, data, col_widths)

        # Grid analysis
        if grid_result:
            pdf.add_page()
            pdf.add_section_title("Grid Analysis")

            pdf.add_key_value(
                "X Spacing", f"{grid_result.get('x_spacing', 'N/A'):.2f} px"
            )
            pdf.add_key_value(
                "Y Spacing", f"{grid_result.get('y_spacing', 'N/A'):.2f} px"
            )
            pdf.ln(3)

            if "grid_image" in grid_result:
                pdf.add_image_from_base64(
                    grid_result["grid_image"],
                    width=160,
                    caption="Calculated grid positions",
                )

        # Histogram analysis
        if histogram_result and "histogram_image" in histogram_result:
            pdf.add_page()
            pdf.add_section_title("Histogram Analysis")
            pdf.add_image_from_base64(
                histogram_result["histogram_image"],
                width=180,
                caption="Pixel value distribution for each grid position",
            )

        # Diagonal comparison
        if diagonal_result:
            pdf.add_page()
            pdf.add_section_title("Diagonal Comparison")

            summary = diagonal_result.get("summary", {})
            pdf.add_subsection_title("Lower Diagonal Statistics")
            pdf.add_key_value(
                "Average Mean", f"{summary.get('lower_avg_mean', 'N/A'):.2f}"
            )
            pdf.add_key_value(
                "Average Median", f"{summary.get('lower_avg_median', 'N/A'):.2f}"
            )
            pdf.add_key_value(
                "Std of Means", f"{summary.get('lower_std_means', 'N/A'):.2f}"
            )
            pdf.ln(3)

            pdf.add_subsection_title("Upper Diagonal Statistics")
            pdf.add_key_value(
                "Average Mean", f"{summary.get('upper_avg_mean', 'N/A'):.2f}"
            )
            pdf.add_key_value(
                "Average Median", f"{summary.get('upper_avg_median', 'N/A'):.2f}"
            )
            pdf.add_key_value(
                "Std of Means", f"{summary.get('upper_std_means', 'N/A'):.2f}"
            )
            pdf.ln(3)

            pdf.add_key_value(
                "Mean Difference", f"{summary.get('mean_difference', 'N/A'):.2f}"
            )
            pdf.ln(5)

            if "comparison_image" in diagonal_result:
                pdf.add_image_from_base64(
                    diagonal_result["comparison_image"],
                    width=180,
                    caption="Lower vs Upper diagonal comparison",
                )

        # Generate PDF bytes
        pdf_bytes = pdf.output()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        return {
            "pdf_base64": pdf_base64,
            "filename": f"circle_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        }

    except Exception as e:
        return {"error": str(e)}


def generate_block_detection_pdf(
    filename,
    result,
    params,
    histogram_result=None,
    subdivision_result=None,
    comparison_result=None,
):
    """
    Generate PDF report for block detection analysis.

    Parameters:
    -----------
    filename : str
        Original filename of the analyzed image
    result : dict
        Block detection results
    params : dict
        Processing parameters used
    histogram_result : dict, optional
        Block histogram analysis results
    subdivision_result : dict, optional
        Subdivision analysis results
    comparison_result : dict, optional
        Block comparison results

    Returns:
    --------
    str : Base64 encoded PDF
    """
    try:
        pdf = ImageAnalysisPDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        # Title section
        pdf.add_section_title("Block Detection Analysis")
        pdf.add_key_value("Source File", filename)
        pdf.add_key_value("Blocks Detected", result.get("count", 0))
        pdf.add_key_value(
            "Total Blocks (with calculated)", len(result.get("all_blocks", []))
        )
        pdf.ln(5)

        # Parameters section
        pdf.add_subsection_title("Processing Parameters")
        pdf.add_key_value("Threshold Value", params.get("threshold_value", "N/A"))
        pdf.add_key_value(
            "Min Length", f"{params.get('min_length_rectangular', 'N/A')} px"
        )
        pdf.add_key_value(
            "Max Length", f"{params.get('max_length_rectangular', 'N/A')} px"
        )
        pdf.add_key_value("Min Rectangularity", params.get("min_rectangularity", "N/A"))
        pdf.add_key_value("Min Solidity", params.get("min_solidity", "N/A"))
        pdf.ln(5)

        # Detection info
        detected_ids = result.get("detected_block_ids", [])
        calculated_ids = result.get("calculated_block_ids", [])
        pdf.add_key_value(
            "Detected Blocks",
            ", ".join(map(str, detected_ids)) if detected_ids else "N/A",
        )
        pdf.add_key_value(
            "Calculated Blocks",
            ", ".join(map(str, calculated_ids)) if calculated_ids else "N/A",
        )

        if result.get("orientation_warning"):
            pdf.ln(3)
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(242, 89, 97)  # Red color
            pdf.multi_cell(0, 5, f"Note: {result['orientation_warning']}")
            pdf.set_text_color(87, 89, 98)
        pdf.ln(5)

        # Detection image
        if "detection_image" in result:
            pdf.add_subsection_title("Detection Result")
            pdf.add_image_from_base64(
                result["detection_image"],
                width=160,
                caption="Detected blocks with labels",
            )

        # Mask image
        if "mask_image" in result:
            pdf.add_subsection_title("Threshold Mask")
            pdf.add_image_from_base64(
                result["mask_image"],
                width=160,
                caption=f"Binary mask (threshold < {params.get('threshold_value', 'N/A')})",
            )

        # Block statistics table
        all_blocks = result.get("all_blocks", [])
        if all_blocks:
            pdf.add_page()
            pdf.add_section_title("Block Statistics")

            headers = ["Block", "Type", "Center (X,Y)", "Width", "Height", "Mean Value"]
            data = []

            for block in all_blocks:
                block_id = block.get("block_id", "N/A")
                block_type = (
                    "Detected" if block.get("detected", False) else "Calculated"
                )
                center = block.get("center", [0, 0])
                center_str = f"({center[0]}, {center[1]})"

                data.append(
                    [
                        str(block_id),
                        block_type,
                        center_str,
                        f"{block.get('width', 'N/A')}",
                        f"{block.get('height', 'N/A')}",
                        (
                            f"{block.get('mean_value', 'N/A'):.1f}"
                            if block.get("mean_value")
                            else "N/A"
                        ),
                    ]
                )

            col_widths = [20, 30, 50, 30, 30, 30]
            pdf.add_table(headers, data, col_widths)

        # Histogram analysis
        if histogram_result and "histogram_image" in histogram_result:
            pdf.add_page()
            pdf.add_section_title("Block Histogram Analysis")
            pdf.add_image_from_base64(
                histogram_result["histogram_image"],
                width=180,
                caption="Pixel value distribution for each block",
            )

        # Subdivision analysis
        if subdivision_result:
            pdf.add_page()
            pdf.add_section_title("Block Subdivisions")

            pdf.add_key_value(
                "Number of Subdivisions",
                subdivision_result.get("num_subdivisions", "N/A"),
            )
            pdf.add_key_value(
                "Total Subdivision Count", subdivision_result.get("total_count", "N/A")
            )
            pdf.ln(3)

            if "subdivision_image" in subdivision_result:
                pdf.add_image_from_base64(
                    subdivision_result["subdivision_image"],
                    width=160,
                    caption="Block subdivisions visualization",
                )

        # Block comparison
        if comparison_result:
            pdf.add_page()
            pdf.add_section_title("Block Comparison Analysis")

            summary = comparison_result.get("summary", {})

            def _fmt_summary_value(key):
                value = summary.get(key)
                return "N/A" if value is None else f"{float(value):.2f}"

            # Block 2 stats
            pdf.add_subsection_title("Block 2 Statistics")
            pdf.add_key_value("Average Mean", _fmt_summary_value("block2_mean_avg"))
            pdf.add_key_value("Std of Means", _fmt_summary_value("block2_mean_std"))
            pdf.ln(2)

            # Block 4 stats
            pdf.add_subsection_title("Block 4 Statistics")
            pdf.add_key_value("Average Mean", _fmt_summary_value("block4_mean_avg"))
            pdf.add_key_value("Std of Means", _fmt_summary_value("block4_mean_std"))
            pdf.ln(2)

            # Comparison
            pdf.add_subsection_title("Comparison")
            pdf.add_key_value(
                "Mean Difference (B2-B4)",
                _fmt_summary_value("coal_difference_avg"),
            )
            pdf.add_key_value("Std of Difference", _fmt_summary_value("coal_difference_std"))
            pdf.ln(5)

            if "comparison_image" in comparison_result:
                pdf.add_image_from_base64(
                    comparison_result["comparison_image"],
                    width=180,
                    caption="Block comparison charts",
                )

        # Generate PDF bytes
        pdf_bytes = pdf.output()
        pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

        return {
            "pdf_base64": pdf_base64,
            "filename": f"block_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        }

    except Exception as e:
        return {"error": str(e)}


# Export functions
__all__ = [
    "generate_circle_detection_pdf",
    "generate_block_detection_pdf",
    "ImageAnalysisPDF",
]
