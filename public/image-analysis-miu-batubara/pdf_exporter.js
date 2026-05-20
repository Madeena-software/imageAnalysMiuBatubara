// JS PDF Exporter aligned with Python `pdf_exporter.py` layout
// Exposes `window.exportCircleResultsToPdf` and `window.exportBlockResultsToPdf`
(async function() {
  function nowStamp() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}${pad(d.getMonth()+1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  }

  async function tryAddImage(pdf, elementId, margin, yPos, contentWidth, maxHeight) {
    if (typeof getImageDataFromElement === 'function') {
      const img = getImageDataFromElement(elementId);
      if (img) {
        const res = await addImageToPdf(pdf, img, margin, yPos, contentWidth, maxHeight);
        return res.height + 8;
      }
    }
    return 0;
  }

  function ensureSpace(pdf, yPos, pageHeight, margin, requiredHeight) {
    if (yPos + requiredHeight > pageHeight - margin) {
      pdf.addPage();
      return margin;
    }
    return yPos;
  }

  function addSectionTitle(pdf, title, margin, yPos) {
    pdf.setFontSize(14);
    pdf.setTextColor(51, 51, 51);
    pdf.text(title, margin, yPos);
    return yPos + 6;
  }

  function addParametersSectionToPdf(pdf, params, margin, yPos, pageHeight) {
    yPos = ensureSpace(pdf, yPos, pageHeight, margin, 18);
    pdf.setFontSize(12);
    pdf.setTextColor(102, 126, 234);
    pdf.text('Processing Parameters', margin, yPos);
    yPos += 7;

    pdf.setFontSize(10);
    pdf.setTextColor(60);
    for (const [key, value] of Object.entries(params)) {
      yPos = ensureSpace(pdf, yPos, pageHeight, margin, 8);
      pdf.setFont('helvetica', 'bold');
      pdf.text(key + ':', margin, yPos);
      pdf.setFont('helvetica', 'normal');
      pdf.text(String(value), margin + 50, yPos, { maxWidth: 120 });
      yPos += 6;
    }
    pdf.setFont('helvetica', 'normal');
    return yPos + 5;
  }

  async function addImageSectionToPdf(pdf, title, elementId, margin, yPos, pageHeight, contentWidth, maxHeight) {
    yPos = ensureSpace(pdf, yPos, pageHeight, margin, maxHeight + 16);
    yPos = addSectionTitle(pdf, title, margin, yPos);
    yPos += await tryAddImage(pdf, elementId, margin, yPos, contentWidth, maxHeight);
    return yPos;
  }

  function addTableSectionToPdf(pdf, tableData, margin, yPos, pageHeight, contentWidth, title) {
    yPos = ensureSpace(pdf, yPos, pageHeight, margin, 28);
    return addTableToPdf(pdf, tableData, margin, yPos, pageHeight, contentWidth, title);
  }

  function getSelectedImageFilename() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput && fileInput.files && fileInput.files.length > 0 && fileInput.files[0].name) {
      return fileInput.files[0].name;
    }
    return 'Not available';
  }

  function getCleanedImageFilename() {
    const name = getSelectedImageFilename();
    if (name === 'Not available') return '';
    return name.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_");
  }

  // Main circle export that mirrors the Python layout requested
  window.exportCircleResultsToPdf = async function() {
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 15;
    const contentWidth = pageWidth - 2 * margin;
    let yPos = margin;

    // Header
    const imageFilename = getSelectedImageFilename();
    pdf.setFontSize(16);
    pdf.setTextColor(21, 114, 232);
    pdf.text('MIU Image Analysis Report', pageWidth / 2, yPos, { align: 'center' });
    yPos += 8;
    pdf.setFontSize(10);
    pdf.setTextColor(100);
    pdf.text('Image File: ' + imageFilename, pageWidth / 2, yPos, { align: 'center', maxWidth: contentWidth });
    yPos += 6;
    pdf.text('Generated: ' + new Date().toLocaleString(), pageWidth / 2, yPos, { align: 'center' });
    yPos += 6;
    pdf.setDrawColor(21, 114, 232);
    pdf.line(margin, yPos, pageWidth - margin, yPos);
    yPos += 8;

    // 1. MIU Summary (try 'circleAttenuationComparison', then 'miuSummary', then fallback to table)
    if (typeof getSummaryTextFromElement === 'function') {
      const miuHtml = getSummaryTextFromElement('circleAttenuationComparison') || 
                      getSummaryTextFromElement('miuSummary');
      if (miuHtml) {
        yPos += 2;
        yPos = ensureSpace(pdf, yPos, pageHeight, margin, 28);
        yPos = addSummarySection(pdf, miuHtml, margin, yPos, pageHeight, 'MIU Summary');
      } else if (typeof getTableDataFromElement === 'function') {
        const diagonalTable = getTableDataFromElement('diagonalSummary');
        if (diagonalTable) {
          yPos = addTableSectionToPdf(pdf, diagonalTable, margin, yPos, pageHeight, contentWidth, 'MIU Summary');
        }
      }
    }

    // 2. Summary Statistics (diagonal summary)
    if (typeof getTableDataFromElement === 'function') {
      const summaryTable = getTableDataFromElement('diagonalSummary');
      if (summaryTable) {
        yPos = addTableSectionToPdf(pdf, summaryTable, margin, yPos, pageHeight, contentWidth, 'Summary Statistics');
      }
    }

    // 2.5 Step-by-Step Calculation
    if (typeof getSummaryTextFromElement === 'function') {
      const stepHtml = getSummaryTextFromElement('stepByStepCalc');
      if (stepHtml) {
        yPos += 4;
        yPos = addSummarySection(pdf, stepHtml, margin, yPos, pageHeight, 'Step-by-Step μ Calculation');
      }
    }

    // 3. Processing Parameters
    if (typeof getParametersFromUI === 'function') {
      const params = getParametersFromUI('circle');
      pdf.addPage();
      yPos = margin;
      yPos = addParametersSectionToPdf(pdf, params, margin, yPos, pageHeight);
      yPos += 6;
    }

    // 4. Detection Result (image)
    yPos = await addImageSectionToPdf(pdf, 'Detection Result', 'detectionImage', margin, yPos, pageHeight, contentWidth, 70);

    // 5. Detection Statistics - All Detected Circles (table)
    if (typeof getTableDataFromElement === 'function') {
      const tableData = getTableDataFromElement('statsTable');
      if (tableData) {
        yPos = addTableSectionToPdf(pdf, tableData, margin, yPos, pageHeight, contentWidth, 'Detection Statistics - All Detected Circles');
      }
    }

    // 6. Grid Analysis (16 Positions)
    yPos = await addImageSectionToPdf(pdf, 'Grid Analysis (16 Positions)', 'gridImage', margin, yPos, pageHeight, contentWidth, 120);

    // 7. Histogram Analysis
    yPos = await addImageSectionToPdf(pdf, 'Histogram Analysis', 'histogramImage', margin, yPos, pageHeight, contentWidth, 160);

    // 8. Histogram Statistics for 16 Circles (try table 'histogramStats' or 'histogramTable')
    if (typeof getTableDataFromElement === 'function') {
      const histTable = getTableDataFromElement('histogramStats') || getTableDataFromElement('histogramTable');
      if (histTable) {
        yPos = addTableSectionToPdf(pdf, histTable, margin, yPos, pageHeight, contentWidth, 'Histogram Statistics for 16 Circles');
      }
    }

    // Footer / save
    if (yPos > pageHeight - 40) { pdf.addPage(); yPos = margin; }
    yPos += 8;
    pdf.setDrawColor(102,126,234);
    pdf.setLineWidth(0.3);
    pdf.line(margin, yPos, pageWidth - margin, yPos);
    yPos += 6;
    pdf.setFontSize(10);
    pdf.setTextColor(100);
    pdf.text('Image Analysis Tool', pageWidth/2, yPos, { align: 'center' });
    yPos += 4;

    const imgName = getCleanedImageFilename();
    const filenamePrefix = imgName ? `circle_detection_report_${imgName}_` : 'circle_detection_report_';
    pdf.save(`${filenamePrefix}${nowStamp()}.pdf`);
  };

  // Basic block exporter that reuses circle layout where appropriate
  window.exportBlockResultsToPdf = async function() {
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const margin = 15;
    const contentWidth = pageWidth - 2 * margin;
    let yPos = margin;

    pdf.setFontSize(16);
    pdf.setTextColor(21, 114, 232);
    pdf.text('Image Analysis Report', pageWidth / 2, yPos, { align: 'center' });
    yPos += 8;
    pdf.setFontSize(10);
    pdf.setTextColor(100);
    pdf.text('Generated: ' + new Date().toLocaleString(), pageWidth / 2, yPos, { align: 'center' });
    yPos += 6;
    pdf.setDrawColor(21, 114, 232);
    pdf.line(margin, yPos, pageWidth - margin, yPos);
    yPos += 8;

    // Parameters
    if (typeof getParametersFromUI === 'function') {
      const params = getParametersFromUI('block');
      pdf.addPage();
      yPos = margin;
      pdf.setFontSize(12);
      pdf.setTextColor(104, 97, 206);
      pdf.text('Processing Parameters', margin, yPos);
      yPos += 6;
      if (typeof addParametersSection === 'function') {
        yPos = addParametersSection(pdf, params, margin, yPos);
      }
      yPos += 6;
    }

    // Detection Result
    pdf.setFontSize(14);
    pdf.setTextColor(51,51,51);
    pdf.text('Detection Result', margin, yPos);
    yPos += 6;
    yPos += await tryAddImage(pdf, 'detectionImage', margin, yPos, contentWidth, 100);

    // Block statistics table
    if (typeof getTableDataFromElement === 'function') {
      const tableData = getTableDataFromElement('statsTable');
      if (tableData) {
        if (yPos > pageHeight - 60) { pdf.addPage(); yPos = margin; }
        yPos = addTableToPdf(pdf, tableData, margin, yPos, pageHeight, contentWidth, 'Block Statistics');
      }
    }

    const imgName = getCleanedImageFilename();
    const filenamePrefix = imgName ? `block_detection_report_${imgName}_` : 'block_detection_report_';
    pdf.save(`${filenamePrefix}${nowStamp()}.pdf`);
  };

})();
