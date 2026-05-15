"use client";
import React, { useState, useEffect } from 'react';
import { FileText, Image as ImageIcon, Wrench, Activity, Wind, AlertTriangle, ShieldAlert, AlertCircle, Info, VolumeX, CheckCircle, Download, Mic } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function ResultsTabs({ data }: { data: any }) {
  const [activeTab, setActiveTab] = useState('Report');
  const [isRecording, setIsRecording] = useState(false);
  const [dictationText, setDictationText] = useState('');
  const [showToast, setShowToast] = useState('');
  const recognitionRef = React.useRef<any>(null);

  const handleDictationToggle = () => {
    if (!('SpeechRecognition' in window) && !(window as any).webkitSpeechRecognition) {
      setShowToast('Speech Recognition not supported');
      return;
    }
    if (isRecording) {
      recognitionRef.current?.stop();
      setIsRecording(false);
      setShowToast('Dictation stopped');
      return;
    }
    const SpeechRecognition = ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition);
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.onresult = (event: any) => {
      let transcript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        transcript += event.results[i][0].transcript;
      }
      setDictationText(prev => prev + transcript);
    };
    recognition.onstart = () => {
      setShowToast('Dictation started');
    };
    recognition.onend = () => {
      setIsRecording(false);
      setShowToast('Dictation ended');
    };
    recognition.onerror = (event: any) => {
      setShowToast('Error: ' + (event as any).error);
      setIsRecording(false);
    };
    recognitionRef.current = recognition;
    recognition.start();
    setIsRecording(true);
  };

  useEffect(() => {
    return () => {
      recognitionRef.current?.abort?.();
    };
  }, []);

  useEffect(() => {
    if (showToast) {
      const timer = setTimeout(() => setShowToast(''), 3000);
      return () => clearTimeout(timer);
    }
  }, [showToast]);



  const tabs = [
    { id: 'Report', icon: <FileText className="w-4 h-4" /> },
    { id: 'Visualizations', icon: <ImageIcon className="w-4 h-4" /> },
    { id: 'Technical', icon: <Wrench className="w-4 h-4" /> },
    { id: 'Anatomy', icon: <Activity className="w-4 h-4" /> },
    { id: 'Pulmonary Patterns', icon: <Wind className="w-4 h-4" /> },
    { id: 'Findings', icon: <AlertTriangle className="w-4 h-4" /> }
  ];

  if (!data) return null;

  const results = data.results;
  const clinicalSummary = results.clinical_summary;

  const handleDownloadText = (filename: string, text: string) => {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDownloadImage = (filename: string, base64: string) => {
    const a = document.createElement('a');
    a.href = base64;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleCompiledDownload = () => {
    let compiled = `AI Chest X-Ray Analysis Report\n`;
    compiled += `===============================\n\n`;
    
    if (data.results.patient_info) {
      const pi = data.results.patient_info;
      compiled += `Patient Information:\n`;
      compiled += `- Name: ${pi.name}\n`;
      compiled += `- MRN: ${pi.mrn}\n`;
      compiled += `- DOB: ${pi.dob}\n`;
      compiled += `- Gender: ${pi.gender}\n`;
      compiled += `- Referring Physician: ${pi.referring_physician}\n\n`;
    }

    compiled += `Comprehensive Report:\n`;
    compiled += `-------------------------------\n`;
    compiled += `${data.results.report || "No report generated."}\n\n`;
    if (dictationText) {
      compiled += `Dictated Notes:\n${dictationText}\n\n`;
    }

    if (clinicalSummary) {
      compiled += `Clinical Findings Summary:\n`;
      compiled += `-------------------------------\n`;
      compiled += `Critical Findings: ${clinicalSummary.clinical_findings.CRITICAL.length}\n`;
      clinicalSummary.clinical_findings.CRITICAL.forEach((i: any) => compiled += `  - ${i.pathology} (${(i.probability*100).toFixed(1)}%)\n`);
      compiled += `\nHigh Priority: ${clinicalSummary.clinical_findings.HIGH.length}\n`;
      clinicalSummary.clinical_findings.HIGH.forEach((i: any) => compiled += `  - ${i.pathology} (${(i.probability*100).toFixed(1)}%)\n`);
      compiled += `\nModerate Findings: ${clinicalSummary.clinical_findings.MODERATE.length}\n`;
      clinicalSummary.clinical_findings.MODERATE.forEach((i: any) => compiled += `  - ${i.pathology} (${(i.probability*100).toFixed(1)}%)\n`);
      compiled += `\n`;
    }

    compiled += `Technical Details:\n`;
    compiled += `-------------------------------\n`;
    compiled += `- View: ${results.view_classification?.view || "N/A"}\n`;
    compiled += `- Rotation: ${results.rotation?.assessment || "N/A"}\n`;
    compiled += `- Penetration: ${results.penetration?.assessment || "N/A"}\n`;

    handleDownloadText(`compiled_xray_report_${Date.now()}.txt`, compiled);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden mt-0 shadow-sm flex flex-col h-full lg:max-h-[850px] min-h-[600px]">
      <div className="flex overflow-x-auto border-b border-slate-200 no-scrollbar bg-slate-50 shrink-0">
        {showToast && (
          <div className="fixed top-4 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-80 text-white px-4 py-2 rounded-md shadow-lg">
            {showToast}
          </div>
        )}
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 flex justify-center items-center gap-2 px-6 py-4 text-sm font-medium whitespace-nowrap transition-colors relative
              ${activeTab === tab.id ? 'text-blue-600' : 'text-slate-500 hover:text-slate-800 hover:bg-slate-100'}`}
          >
            {tab.icon}
            {tab.id}
            {activeTab === tab.id && (
              <motion.div layoutId="activeTab" className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600" />
            )}
          </button>
        ))}
        <div className="flex-1"></div>
        <button 
          onClick={handleCompiledDownload}
          className="flex items-center justify-center gap-2 px-6 py-4 text-sm font-bold whitespace-nowrap text-blue-600 hover:text-blue-700 hover:bg-blue-100 transition-colors border-l border-slate-200"
        >
          <Download className="w-4 h-4" />
          Compiled Report
        </button>
      </div>
      
      <div className="p-6 flex-1 flex flex-col min-h-0 overflow-y-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="h-full flex flex-col"
          >
            {activeTab === 'Report' && (
              <div className="h-full flex flex-col min-h-0">
                <div className="flex justify-between items-center mb-4 shrink-0">
                  <h3 className="text-xl font-bold text-slate-800">Comprehensive Radiology Report</h3>
                  <button 
                    onClick={handleDictationToggle}
                    className="flex items-center gap-2 px-4 py-2 bg-blue-100 hover:bg-blue-200 text-blue-800 rounded-lg text-sm font-medium transition-colors border border-blue-200"
                  >
                    <Mic className="w-4 h-4" /> {isRecording ? 'Stop Dictation' : 'Start Dictation'}
                  </button>
                  <button 
                    onClick={() => {
                      const baseReport = results.report || '';
                      const finalReport = dictationText ? `${baseReport}\n\nDictated Notes:\n${dictationText}` : baseReport;
                      handleDownloadText('radiology_report.txt', finalReport);
                    }}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium transition-colors border border-slate-200"
                  >
                    <Download className="w-4 h-4" /> Download Text
                  </button>
                </div>
                <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 whitespace-pre-wrap font-mono text-sm text-slate-700 flex-1 overflow-y-auto leading-relaxed shadow-inner">
                  {results.report || "No report available."}
                  <textarea
                    value={dictationText}
                    onChange={(e) => setDictationText(e.target.value)}
                    placeholder="Dictated notes will appear here..."
                    className="mt-4 w-full h-32 p-2 border border-slate-300 rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            )}
            
            {activeTab === 'Visualizations' && (
              <div className="h-full flex flex-col min-h-0">
                <h3 className="text-xl font-bold text-slate-800 mb-4 shrink-0">Anatomical Segmentation & Measurements</h3>
                <div className="flex flex-col gap-6 flex-1 min-h-0 overflow-y-auto pr-2 pb-4">
                  <div className="flex flex-col shrink-0 h-[200px] sm:h-[250px]">
                    <div className="flex justify-between items-center mb-2 shrink-0">
                      <p className="text-sm text-slate-500 font-medium">Original X-Ray</p>
                      <button 
                        onClick={() => handleDownloadImage('original_xray.png', data.original_base64)}
                        className="text-xs flex items-center gap-1 text-blue-600 hover:text-blue-800 px-2 py-1 bg-blue-50 hover:bg-blue-100 rounded transition-colors"
                      >
                        <Download className="w-3 h-3" /> Save
                      </button>
                    </div>
                    <div className="flex-1 bg-slate-50 rounded-xl border border-slate-200 overflow-hidden relative">
                      <img src={data.original_base64} alt="Original" className="absolute inset-0 w-full h-full object-contain p-2" />
                    </div>
                  </div>
                  <div className="flex flex-col flex-1 min-h-[400px] sm:min-h-[600px]">
                    <div className="flex justify-between items-center mb-2 shrink-0">
                      <p className="text-sm text-slate-500 font-medium">Annotated Analysis</p>
                      {data.visualization_base64 && (
                        <button 
                          onClick={() => handleDownloadImage('annotated_xray.png', data.visualization_base64)}
                          className="text-xs flex items-center gap-1 text-blue-600 hover:text-blue-800 px-2 py-1 bg-blue-50 hover:bg-blue-100 rounded transition-colors"
                        >
                          <Download className="w-3 h-3" /> Save
                        </button>
                      )}
                    </div>
                    {data.visualization_base64 ? (
                      <div className="flex-1 bg-slate-50 rounded-xl border border-slate-200 overflow-hidden relative">
                        <img src={data.visualization_base64} alt="Visualization" className="absolute inset-0 w-full h-full object-contain p-2" />
                      </div>
                    ) : (
                      <div className="flex-1 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-center text-slate-400">Visualization not generated</div>
                    )}
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === 'Technical' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 overflow-y-auto pr-2">
                <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 h-fit">
                  <h4 className="text-lg font-bold text-slate-800 mb-4 border-b border-slate-200 pb-2">Projection</h4>
                  <div className="flex justify-between items-center bg-white p-3 rounded-lg border border-slate-200">
                    <span className="text-slate-500">View</span>
                    <span className="font-semibold text-slate-800">{results.view_classification?.view || "N/A"}</span>
                  </div>
                  
                  <h4 className="text-lg font-bold text-slate-800 mt-6 mb-4 border-b border-slate-200 pb-2">Rotation</h4>
                  <div className={`p-4 rounded-lg font-medium border ${results.rotation?.symmetric ? 'bg-green-50 border-green-200 text-green-700' : 'bg-amber-50 border-amber-200 text-amber-700'}`}>
                    {results.rotation?.assessment || "Rotation assessment unavailable"}
                  </div>
                </div>
                
                <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 h-fit">
                  <h4 className="text-lg font-bold text-slate-800 mb-4 border-b border-slate-200 pb-2">Inspiration</h4>
                  <div className="flex justify-between items-center bg-white p-3 rounded-lg border border-slate-200">
                    <span className="text-slate-500">Posterior Ribs Visible</span>
                    <span className="font-bold text-2xl text-slate-800">{results.inspiration?.count || 0}</span>
                  </div>
                  
                  <h4 className="text-lg font-bold text-slate-800 mt-6 mb-4 border-b border-slate-200 pb-2">Penetration</h4>
                  <div className={`p-4 rounded-lg font-medium border ${results.penetration?.adequate ? 'bg-green-50 border-green-200 text-green-700' : 'bg-amber-50 border-amber-200 text-amber-700'}`}>
                    {results.penetration?.assessment || "Check penetration"}
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === 'Anatomy' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 overflow-y-auto pr-2">
                <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 h-fit">
                  <h4 className="text-lg font-bold text-slate-800 mb-4 border-b border-slate-200 pb-2">Airway</h4>
                  <div className={`p-4 rounded-lg font-medium border ${results.trachea?.midline ? 'bg-green-50 border-green-200 text-green-700' : 'bg-amber-50 border-amber-200 text-amber-700'}`}>
                    {results.trachea?.assessment || "Status unknown"}
                  </div>
                </div>
                
                <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 h-fit">
                  <h4 className="text-lg font-bold text-slate-800 mb-4 border-b border-slate-200 pb-2">Cardiac Silhouette</h4>
                  {results.ctr ? (
                    <>
                      <div className="flex justify-between items-center bg-white p-3 rounded-lg border border-slate-200 mb-4">
                        <span className="text-slate-500">CTR Value</span>
                        <span className="font-bold text-2xl text-slate-800">{results.ctr.ctr ? Number(results.ctr.ctr).toFixed(3) : "N/A"}</span>
                      </div>
                      <div className={`p-4 rounded-lg font-medium border ${results.ctr.ctr < 0.5 ? 'bg-green-50 border-green-200 text-green-700' : 'bg-amber-50 border-amber-200 text-amber-700'}`}>
                        {results.ctr.assessment || "Normal"}
                      </div>
                    </>
                  ) : (
                    <div className="text-slate-500 italic">No CTR data available.</div>
                  )}
                </div>
              </div>
            )}
            
            {activeTab === 'Pulmonary Patterns' && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 overflow-y-auto pr-2">
                <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 h-fit">
                  <h4 className="text-lg font-bold text-slate-800 mb-4 border-b border-slate-200 pb-2">Edema Assessment</h4>
                  {results.edema_pattern ? (
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-slate-500">Pattern</span>
                        <span className="font-semibold text-slate-800">{results.edema_pattern.pattern_type}</span>
                      </div>
                      <div className="flex justify-between mb-4">
                        <span className="text-slate-500">Confidence</span>
                        <span className="font-semibold text-slate-500">{(results.edema_pattern.confidence || 0) * 100}%</span>
                      </div>
                      {results.edema_pattern.supporting_features?.length > 0 && (
                        <div className="mt-4">
                          <p className="text-sm font-medium text-slate-700 mb-2">Features:</p>
                          <ul className="list-disc list-inside text-sm text-slate-600 space-y-1">
                            {results.edema_pattern.supporting_features.map((f: string, i: number) => <li key={i}>{f}</li>)}
                          </ul>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-slate-500 italic">Not evaluated</div>
                  )}
                </div>
                
                <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 h-fit">
                  <h4 className="text-lg font-bold text-slate-800 mb-4 border-b border-slate-200 pb-2">Pulmonary Fissures</h4>
                  {results.fissures ? (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 text-slate-600">
                        <span className="w-2 h-2 rounded-full bg-slate-400"></span>
                        {results.fissures.major_fissures?.length || 0} major fissure(s) detected
                      </div>
                      {results.fissures.minor_fissure && (
                        <div className="flex items-center gap-2 text-slate-600">
                          <span className="w-2 h-2 rounded-full bg-slate-400"></span>
                          Minor fissure detected
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-slate-500 italic">No fissures detected or evaluated</div>
                  )}
                </div>
              </div>
            )}
            
            {activeTab === 'Findings' && (
              <div className="space-y-6 overflow-y-auto pr-2 h-full">
                {!clinicalSummary ? (
                  <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 text-slate-500 italic text-center">
                    Clinical summary not available. Model might not be fully loaded.
                  </div>
                ) : (
                  <>
                    {/* Summary Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-white border border-slate-200 p-4 rounded-xl flex flex-col items-center justify-center text-center shadow-sm">
                        <ShieldAlert className="w-8 h-8 text-red-600 mb-2" />
                        <span className="text-2xl font-bold text-slate-800">{clinicalSummary.clinical_findings.CRITICAL.length}</span>
                        <span className="text-xs text-red-600 uppercase tracking-wider font-semibold">Critical</span>
                      </div>
                      <div className="bg-white border border-slate-200 p-4 rounded-xl flex flex-col items-center justify-center text-center shadow-sm">
                        <AlertCircle className="w-8 h-8 text-orange-500 mb-2" />
                        <span className="text-2xl font-bold text-slate-800">{clinicalSummary.clinical_findings.HIGH.length}</span>
                        <span className="text-xs text-orange-500 uppercase tracking-wider font-semibold">High Priority</span>
                      </div>
                      <div className="bg-white border border-slate-200 p-4 rounded-xl flex flex-col items-center justify-center text-center shadow-sm">
                        <Info className="w-8 h-8 text-amber-500 mb-2" />
                        <span className="text-2xl font-bold text-slate-800">{clinicalSummary.clinical_findings.MODERATE.length}</span>
                        <span className="text-xs text-amber-500 uppercase tracking-wider font-semibold">Moderate</span>
                      </div>
                      <div className="bg-white border border-slate-200 p-4 rounded-xl flex flex-col items-center justify-center text-center shadow-sm">
                        <VolumeX className="w-8 h-8 text-slate-400 mb-2" />
                        <span className="text-2xl font-bold text-slate-800">{clinicalSummary.total_suppressed}</span>
                        <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">Suppressed</span>
                      </div>
                    </div>

                    {/* Critical Findings */}
                    {clinicalSummary.clinical_findings.CRITICAL.length > 0 && (
                      <div className="bg-red-50 border border-red-200 p-6 rounded-xl">
                        <h4 className="text-red-700 font-bold mb-4 flex items-center gap-2 uppercase tracking-wider text-sm">
                          <ShieldAlert className="w-5 h-5" /> Immediate Attention Required
                        </h4>
                        <div className="space-y-3">
                          {clinicalSummary.clinical_findings.CRITICAL.map((item: any, i: number) => (
                            <div key={i} className="bg-white p-4 rounded-lg border border-red-200 shadow-sm">
                              <div className="flex justify-between items-start">
                                <div>
                                  <span className="text-lg font-bold text-slate-800">{item.pathology}</span>
                                  <p className="text-sm text-red-600 mt-1">Analysis demonstrates this finding.</p>
                                </div>
                                <div className="text-right">
                                  <span className="bg-red-100 text-red-700 text-xs px-2 py-1 rounded font-bold">
                                    {(item.probability * 100).toFixed(1)}% Confidence
                                  </span>
                                  <p className="text-xs text-slate-500 mt-1">Threshold: {(item.threshold_used * 100).toFixed(0)}%</p>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* High Priority Findings */}
                    {clinicalSummary.clinical_findings.HIGH.length > 0 && (
                      <div className="bg-orange-50 border border-orange-200 p-6 rounded-xl">
                        <h4 className="text-orange-700 font-bold mb-4 flex items-center gap-2 uppercase tracking-wider text-sm">
                          <AlertCircle className="w-5 h-5" /> Significant Clinical Findings
                        </h4>
                        <div className="space-y-3">
                          {clinicalSummary.clinical_findings.HIGH.map((item: any, i: number) => (
                            <div key={i} className="bg-white p-4 rounded-lg border border-orange-200 shadow-sm">
                              <div className="flex justify-between items-start">
                                <div>
                                  <span className="text-lg font-bold text-slate-800">{item.pathology}</span>
                                  <p className="text-sm text-orange-600 mt-1">Recommend prompt clinical evaluation.</p>
                                </div>
                                <div className="text-right">
                                  <span className="bg-orange-100 text-orange-700 text-xs px-2 py-1 rounded font-bold">
                                    {(item.probability * 100).toFixed(1)}% Confidence
                                  </span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Moderate Findings */}
                    {clinicalSummary.clinical_findings.MODERATE.length > 0 && (
                      <div className="bg-amber-50 border border-amber-200 p-6 rounded-xl">
                        <h4 className="text-amber-700 font-bold mb-4 flex items-center gap-2 uppercase tracking-wider text-sm">
                          <Info className="w-5 h-5" /> Notable Clinical Findings
                        </h4>
                        <div className="space-y-3">
                          {clinicalSummary.clinical_findings.MODERATE.map((item: any, i: number) => (
                            <div key={i} className="bg-white p-4 rounded-lg border border-amber-200 shadow-sm">
                              <div className="flex justify-between items-center">
                                <span className="font-bold text-slate-800">{item.pathology}</span>
                                <span className="text-amber-600 font-mono text-sm">{(item.probability * 100).toFixed(1)}%</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Suppressed / Low / Negative Summary if no major issues */}
                    {clinicalSummary.total_positive_findings === 0 && (
                       <div className="p-8 text-center bg-green-50 border border-green-200 rounded-xl flex flex-col items-center mb-4">
                          <CheckCircle className="w-12 h-12 text-green-500 mb-3" />
                          <h4 className="text-green-700 font-bold text-lg">No Significant Abnormalities Detected</h4>
                          <p className="text-sm text-green-600 mt-2 max-w-md">The rule-based filter suppressed or found no high-confidence evidence for critical pathologies.</p>
                       </div>
                    )}

                    {/* Suppressed Findings List (Collapsible or small) */}
                    {clinicalSummary.clinical_findings.SUPPRESSED.length > 0 && (
                      <div className="mt-8 border-t border-slate-200 pt-6 pb-6">
                        <h4 className="text-slate-500 font-semibold mb-4 text-sm flex items-center gap-2">
                          <VolumeX className="w-4 h-4" /> Rule-Filtered / Suppressed Findings
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {clinicalSummary.clinical_findings.SUPPRESSED.map((item: any, i: number) => (
                            <div key={i} className="bg-slate-50 p-3 rounded-lg border border-slate-200 flex justify-between items-center text-sm shadow-sm">
                              <span className="text-slate-400 line-through decoration-slate-300">{item.pathology}</span>
                              <span className="text-xs text-slate-500 italic max-w-[150px] truncate" title={item.suppression_reason}>{item.suppression_reason}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
