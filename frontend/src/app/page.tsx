"use client";

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Server } from 'lucide-react';
import FileUploader from '@/components/FileUploader';
import PatientForm from '@/components/PatientForm';
import ResultsTabs from '@/components/ResultsTabs';

export default function Home() {
  const [status, setStatus] = useState<any>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [resultsData, setResultsData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    name: '', mrn: '', dob: '', gender: 'Male', referring_physician: '', history: ''
  });

  useEffect(() => {
    const checkStatus = () => {
      axios.get('http://localhost:8000/api/status')
        .then(res => setStatus(res.data.status))
        .catch(err => {
          console.error("Backend not reachable", err);
          setStatus(null);
        });
    };

    // Initial check
    checkStatus();

    // Poll every 5 seconds to auto-update
    const intervalId = setInterval(checkStatus, 5000);

    return () => clearInterval(intervalId);
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;
    
    setIsAnalyzing(true);
    setError(null);
    setResultsData(null);
    
    const data = new FormData();
    data.append('file', file);
    data.append('name', formData.name);
    data.append('mrn', formData.mrn);
    data.append('dob', formData.dob);
    data.append('gender', formData.gender);
    data.append('referring_physician', formData.referring_physician);
    data.append('history', formData.history);

    try {
      const res = await axios.post('http://localhost:8000/api/analyze', data, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResultsData(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "An error occurred during analysis.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 p-4 md:p-8 font-sans selection:bg-blue-500/30">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-200 pb-6">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
              <span className="p-2 bg-blue-600 rounded-lg text-white shadow-lg shadow-blue-500/20"><Activity className="w-6 h-6" /></span>
              AI Chest X-Ray Analyzer
            </h1>
            <p className="text-slate-500 mt-2">Comprehensive automated analysis of chest X-rays</p>
          </div>
          
          <div className="flex items-center gap-3 bg-white border border-slate-200 rounded-xl p-3 shadow-sm">
            <Server className="w-5 h-5 text-slate-400" />
            <div className="text-sm font-medium">
              <div className="text-slate-500">Backend Status</div>
              {status ? (
                <div className="flex gap-4 mt-1">
                  <span className={`flex items-center gap-1 ${status.view_classifier ? 'text-green-600' : 'text-red-600'}`}>
                    <span className={`w-2 h-2 rounded-full ${status.view_classifier ? 'bg-green-500' : 'bg-red-500'}`}></span>
                    Classifier
                  </span>
                  <span className={`flex items-center gap-1 ${status.report_generator ? 'text-green-600' : 'text-amber-600'}`}>
                    <span className={`w-2 h-2 rounded-full ${status.report_generator ? 'bg-green-500' : 'bg-amber-500'}`}></span>
                    Generator
                  </span>
                </div>
              ) : (
                <span className="text-amber-500 animate-pulse">Connecting...</span>
              )}
            </div>
          </div>
        </header>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Left Column: Upload & Patient Info */}
          <div className="lg:col-span-1 flex flex-col gap-6">
            <FileUploader onFileSelect={setFile} />
            <PatientForm formData={formData} setFormData={setFormData} />
            
            <button 
              onClick={handleAnalyze}
              disabled={!file || isAnalyzing}
              className={`w-full py-4 rounded-xl font-bold text-lg transition-all transform flex justify-center items-center gap-2 shadow-sm
                ${!file || isAnalyzing 
                  ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200' 
                  : 'bg-blue-600 hover:bg-blue-500 text-white hover:scale-[1.02] border border-blue-500 shadow-blue-500/20'}`}
            >
              {isAnalyzing ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing Image...
                </>
              ) : (
                "Analyze X-Ray"
              )}
            </button>
            
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-600 p-4 rounded-xl font-medium shadow-sm">
                {error}
              </div>
            )}
            
            {/* Medical Disclaimer */}
            <div className="bg-white border border-slate-200 p-4 rounded-xl mt-4 shadow-sm">
              <p className="text-xs text-slate-500 text-center uppercase tracking-wider font-semibold mb-1">Medical Disclaimer</p>
              <p className="text-xs text-slate-500 text-center">
                This is an AI-assisted tool. All findings must be verified by a qualified radiologist. Not for emergency medical use.
              </p>
            </div>
          </div>
          
          {/* Right Column: Results */}
          <div className="lg:col-span-2">
            {!resultsData && !isAnalyzing && (
              <div className="h-full min-h-[500px] border-2 border-dashed border-slate-300 rounded-3xl flex flex-col items-center justify-center text-slate-400 bg-white p-8 shadow-sm">
                <Activity className="w-16 h-16 mb-4 opacity-50 text-slate-400" />
                <h3 className="text-xl font-medium text-slate-800">Awaiting Analysis</h3>
                <p className="text-sm mt-2 max-w-sm text-center text-slate-500">Upload a chest X-Ray and fill out the patient information to begin the comprehensive automated analysis.</p>
              </div>
            )}
            
            {isAnalyzing && (
              <div className="h-full min-h-[500px] border border-slate-200 rounded-3xl flex flex-col items-center justify-center text-slate-500 bg-white shadow-sm">
                <div className="relative mb-8">
                  <div className="w-24 h-24 rounded-full border-t-2 border-r-2 border-blue-600 animate-spin"></div>
                  <div className="w-24 h-24 rounded-full border-b-2 border-l-2 border-slate-200 animate-spin absolute top-0 left-0" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}></div>
                </div>
                <h3 className="text-xl font-medium animate-pulse text-slate-800">Running ML Pipeline...</h3>
                <p className="text-sm mt-2 text-slate-500">Applying advanced clinical rule-based filtering...</p>
              </div>
            )}
            
            {resultsData && !isAnalyzing && (
              <div className="h-full">
                <ResultsTabs data={resultsData} />
              </div>
            )}
          </div>
          
        </div>
      </div>
    </div>
  );
}
