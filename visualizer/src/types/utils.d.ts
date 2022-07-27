export type ValueOf<T> = T extends any[] ? T[number] : T[keyof T]

export type Nullable<T> = T | null
export type Undefinable<T> = T | undefined