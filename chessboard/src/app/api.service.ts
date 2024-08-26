import { Injectable } from '@angular/core';
import { HttpClient, HttpParams  } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private apiUrl= environment.apiUrl;

  constructor(private http: HttpClient) { }

  getData(endpoint: string, model: string, fen?: string): Observable<any> {
    let httpParams = new HttpParams();
    if (fen) {
      httpParams = httpParams.append('fen', fen);
    }

    return this.http.get<any>(`${this.apiUrl}/${endpoint}/${model}`, { params: httpParams });
  }
}