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

    // 1. Summary MIU (try 'miuSummary' then 'diagonalSummary')
    if (typeof getSummaryTextFromElement === 'function') {
      const miuHtml = document.getElementById('miuSummary') ? document.getElementById('miuSummary').innerText : null;
      if (miuHtml) {
        yPos += 2;
        yPos = addSummarySection(pdf, miuHtml, margin, yPos, pageHeight, 'MIU Summary');
      } else {
        const diagonalTable = typeof getTableDataFromElement === 'function' && getTableDataFromElement('diagonalSummary');
        if (diagonalTable) {
          yPos = addTableToPdf(pdf, diagonalTable, margin, yPos, pageHeight, contentWidth, 'MIU Summary');
        }
      }
    }

    // 2. Summary Statistics (diagonal summary)
    if (typeof getTableDataFromElement === 'function') {
      const summaryTable = getTableDataFromElement('diagonalSummary');
      if (summaryTable) {
        if (yPos > pageHeight - 80) { pdf.addPage(); yPos = margin; }
        yPos = addTableToPdf(pdf, summaryTable, margin, yPos, pageHeight, contentWidth, 'Summary Statistics');
      }
    }

    // 3. Parameters
    if (typeof getParametersFromUI === 'function') {
      const params = getParametersFromUI('circle');
      if (yPos > pageHeight - 90) { pdf.addPage(); yPos = margin; }
      pdf.setFontSize(12);
      pdf.setTextColor(104, 97, 206);
      pdf.text('Processing Parameters', margin, yPos);
      yPos += 6;
      if (typeof addParametersSection === 'function') {
        yPos = addParametersSection(pdf, params, margin, yPos);
      } else {
        pdf.setFontSize(10);
        pdf.setTextColor(60);
        for (const k in params) {
          pdf.setFont('helvetica', 'bold');
          pdf.text(k + ':', margin, yPos);
          pdf.setFont('helvetica', 'normal');
          pdf.text(String(params[k]), margin + 50, yPos);
          yPos += 5;
          if (yPos > pageHeight - 40) { pdf.addPage(); yPos = margin; }
        }
      }
      yPos += 6;
    }

    // 4. Detection Result (image)
    pdf.setFontSize(14);
    pdf.setTextColor(51,51,51);
    pdf.text('Detection Result', margin, yPos);
    yPos += 6;
    yPos += await tryAddImage(pdf, 'detectionImage', margin, yPos, contentWidth, 70);

    // 5. Detection Statistics - All Detected Circles (table)
    if (typeof getTableDataFromElement === 'function') {
      const tableData = getTableDataFromElement('statsTable');
      if (tableData) {
        if (yPos > pageHeight - 60) { pdf.addPage(); yPos = margin; }
        yPos = addTableToPdf(pdf, tableData, margin, yPos, pageHeight, contentWidth, 'Detection Statistics - All Detected Circles');
      }
    }

    // 6. Grid Analysis (16 Positions)
    if (yPos > pageHeight - 120) { pdf.addPage(); yPos = margin; }
    pdf.setFontSize(14);
    pdf.setTextColor(51,51,51);
    pdf.text('Grid Analysis (16 Positions)', margin, yPos);
    yPos += 6;
    yPos += await tryAddImage(pdf, 'gridImage', margin, yPos, contentWidth, 120);

    // 7. Histogram analysis
    if (yPos > pageHeight - 160) { pdf.addPage(); yPos = margin; }
    pdf.setFontSize(14);
    pdf.setTextColor(51,51,51);
    pdf.text('Histogram Analysis', margin, yPos);
    yPos += 6;
    yPos += await tryAddImage(pdf, 'histogramImage', margin, yPos, contentWidth, 160);

    // 8. Histogram Statistics for 16 Circles (try table 'histogramStats' or 'histogramTable')
    if (typeof getTableDataFromElement === 'function') {
      const histTable = getTableDataFromElement('histogramStats') || getTableDataFromElement('histogramTable');
      if (histTable) {
        if (yPos > pageHeight - 100) { pdf.addPage(); yPos = margin; }
        yPos = addTableToPdf(pdf, histTable, margin, yPos, pageHeight, contentWidth, 'Histogram Statistics for 16 Circles');
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
    pdf.text('Image Analysis Tool - Circle & Block Detection', pageWidth/2, yPos, { align: 'center' });
    yPos += 4;

    pdf.save(`circle_detection_report_${nowStamp()}.pdf`);
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

    pdf.save(`block_detection_report_${nowStamp()}.pdf`);
  };

})();
