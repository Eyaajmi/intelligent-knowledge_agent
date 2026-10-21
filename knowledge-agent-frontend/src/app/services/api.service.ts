import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { API_URL } from '../config';

export interface DocumentInfo {
  id: string;
  name: string;
  chunks: number;
  pages: number;
}

export interface DocumentsResponse {
  documents: DocumentInfo[];
}

export interface SourceRef {
  source: string;
  name: string;
  page: number;
  snippet: string;
  score: number | null;
}

export interface AskTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface AskRequest {
  question: string;
  source: string | null;
  history: AskTurn[];
}

export interface UploadResponse {
  message: string;
  id: string;
  name: string;
  pages: number;
  chunks_added: number;
}

export type StreamEvent =
  | { type: 'meta'; tool: string; answerable: boolean; sources: SourceRef[]; rewritten_question: string | null }
  | { type: 'token'; text: string }
  | { type: 'done' }
  | { type: 'error'; detail: string };

/** Erreur lisible par l'utilisateur (message du backend si disponible). */
export class ApiError extends Error {
  constructor(message: string, public status?: number) {
    super(message);
  }
}

/** FastAPI renvoie `detail` en texte (HTTPException) ou en liste (validation 422). */
export function detailToMessage(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') {
    return detail;
  }
  if (Array.isArray(detail)) {
    const msgs = detail.map((d) => (d && typeof d.msg === 'string' ? d.msg : '')).filter(Boolean);
    if (msgs.length) {
      return msgs.join(' ');
    }
  }
  return fallback;
}

export function describeError(err: unknown): string {
  if (err instanceof ApiError) {
    return err.message;
  }
  if (err instanceof HttpErrorResponse) {
    if (err.status === 0) {
      return `Impossible de joindre le serveur (${API_URL}). Est-il démarré ?`;
    }
    return detailToMessage(err.error?.detail, `Erreur ${err.status} du serveur.`);
  }
  if (err instanceof TypeError) {
    return `Impossible de joindre le serveur (${API_URL}). Est-il démarré ?`;
  }
  return 'Une erreur inattendue est survenue.';
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly apiUrl = API_URL;

  constructor(private http: HttpClient) {}

  getDocuments(): Observable<DocumentsResponse> {
    return this.http.get<DocumentsResponse>(`${this.apiUrl}/documents`);
  }

  uploadDocument(file: File): Observable<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<UploadResponse>(`${this.apiUrl}/documents/upload`, formData);
  }

  deleteDocument(source: string): Observable<unknown> {
    return this.http.delete(`${this.apiUrl}/documents`, { params: { source } });
  }

  /** URL qui ouvre le PDF à la page voulue dans le lecteur du navigateur. */
  pageUrl(source: string, page: number): string {
    return `${this.apiUrl}/documents/file?source=${encodeURIComponent(source)}#page=${page}`;
  }

  /**
   * Pose une question et reçoit la réponse en flux (Server-Sent Events).
   * HttpClient ne sait pas lire un flux : on utilise fetch.
   */
  async askStream(body: AskRequest, onEvent: (event: StreamEvent) => void, signal?: AbortSignal): Promise<void> {
    const response = await fetch(`${this.apiUrl}/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });

    if (!response.ok) {
      let detail: unknown;
      try {
        detail = (await response.json()).detail;
      } catch {
        /* corps non JSON */
      }
      throw new ApiError(detailToMessage(detail, `Erreur ${response.status} du serveur.`), response.status);
    }
    if (!response.body) {
      throw new ApiError('Le serveur n’a renvoyé aucun flux.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      let end: number;
      while ((end = buffer.indexOf('\n\n')) >= 0) {
        const raw = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        if (raw.startsWith('data: ')) {
          onEvent(JSON.parse(raw.slice(6)) as StreamEvent);
        }
      }
    }
  }
}
