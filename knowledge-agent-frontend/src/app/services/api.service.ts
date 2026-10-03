import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface AskResponse {
  answer: string;
  sources: {
    page: number;
    source: string;
  }[];
}
export interface DocumentsResponse {
  documents: string[];
}
@Injectable({
  providedIn: 'root'
})

export class ApiService {

  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  askQuestion(
  question: string,
  source?: string
): Observable<AskResponse> {

  return this.http.post<AskResponse>(
    `${this.apiUrl}/ask`,
    {
      question: question,
      source: source
    }
  );
}

  uploadDocument(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);

    return this.http.post(
      `${this.apiUrl}/documents/upload`,
      formData
    );
  }
  getDocuments(): Observable<DocumentsResponse> {
  return this.http.get<DocumentsResponse>(
    `${this.apiUrl}/documents`
  );
}
}

